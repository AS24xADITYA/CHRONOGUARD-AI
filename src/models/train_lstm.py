"""ChronoGuard LSTM Forecaster Training Pipeline.

Implements config-driven training of the 2-layer LSTM + Self-Attention Forecaster
per AI-INSTRUCTIONS/01-ai-agent-instructions.md Phase 3 and AI-INSTRUCTIONS/05-ml-algorithms.md.

Saves:
- models/chronoguard_lstm.pt
- results/lstm_metrics.json
- results/training_curves.png
"""

import os
import json
import yaml
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.models.lstm_model import ChronoGuardLSTM
from src.data.load_raw import load_raw_dataset
from src.data.windowing import (
    bucket_flows_into_windows,
    create_sequences,
)
from src.data.label_mapping import STAGE_NAMES


def set_seed(seed: int = 42):
    """Ensure end-to-end reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_lstm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forecast_k: int = 3,
) -> Dict[str, Any]:
    """Evaluate model on held-out test split."""
    model.eval()
    all_prob_preds = []
    all_prob_targets = []
    all_stage_preds = []
    all_stage_targets = []

    with torch.no_grad():
        for batch_x, batch_prob, batch_stage in loader:
            batch_x = batch_x.to(device)
            prob_pred, stage_logits, _ = model(batch_x)
            
            all_prob_preds.append(prob_pred.cpu().numpy())
            all_prob_targets.append(batch_prob.numpy())
            
            stage_pred = torch.argmax(stage_logits, dim=-1).cpu().numpy()
            all_stage_preds.append(stage_pred)
            all_stage_targets.append(batch_stage.numpy())

    prob_preds = np.concatenate(all_prob_preds, axis=0)
    prob_targets = np.concatenate(all_prob_targets, axis=0)
    stage_preds = np.concatenate(all_stage_preds, axis=0)
    stage_targets = np.concatenate(all_stage_targets, axis=0)

    # Standard metrics on stage prediction
    acc = float(accuracy_score(stage_targets, stage_preds))
    prec = float(precision_score(stage_targets, stage_preds, average="macro", zero_division=0))
    rec = float(recall_score(stage_targets, stage_preds, average="macro", zero_division=0))
    f1 = float(f1_score(stage_targets, stage_preds, average="macro", zero_division=0))

    # Binary False Positive Rate: Benign (class 0) vs Attack (classes 1..5)
    binary_true = (stage_targets > 0).astype(int)
    binary_pred = (stage_preds > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Per-horizon K-step forecast accuracy (evaluating attack probability threshold >= 0.5)
    k_step_metrics = {}
    for step in range(forecast_k):
        step_true = (prob_targets[:, step] >= 0.5).astype(int)
        step_pred = (prob_preds[:, step] >= 0.5).astype(int)
        step_acc = float(accuracy_score(step_true, step_pred))
        step_f1 = float(f1_score(step_true, step_pred, average="binary", zero_division=0))
        k_step_metrics[f"horizon_t+{step+1}"] = {
            "accuracy": round(step_acc, 4),
            "f1_score": round(step_f1, 4),
        }

    stage_cm = confusion_matrix(
        stage_targets,
        stage_preds,
        labels=list(range(len(STAGE_NAMES))),
    ).tolist()

    report = classification_report(
        stage_targets,
        stage_preds,
        labels=list(range(len(STAGE_NAMES))),
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": "ChronoGuard LSTM Forecaster",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "total_test_samples": int(len(stage_targets)),
        "k_step_forecast": k_step_metrics,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "stage_confusion_matrix": {
            "classes": STAGE_NAMES,
            "matrix": stage_cm,
        },
        "per_stage": {
            stage: {
                "precision": round(report[stage]["precision"], 4),
                "recall": round(report[stage]["recall"], 4),
                "f1_score": round(report[stage]["f1-score"], 4),
                "support": int(report[stage]["support"]),
            }
            for stage in STAGE_NAMES if stage in report
        },
    }


def train_lstm(
    config_path: str = "config.yaml",
    sample_per_file: int = 5000,
) -> Dict[str, Any]:
    """Execute complete config-driven LSTM training workflow."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["model"]["random_seed"])
    device = torch.device("cpu")  # Strictly CPU per zero-cost constraints

    raw_dir = cfg["dataset"]["raw_dir"]
    sample_per_file = cfg.get("dataset", {}).get("sample_per_file", sample_per_file)
    selected_features = cfg["dataset"]["selected_features"]
    window_size_flows = cfg["windowing"]["window_size_flows"]
    seq_w = cfg["windowing"]["sequence_length_w"]
    forecast_k = cfg["windowing"]["forecast_horizon_k"]

    batch_size = cfg["model"]["batch_size"]
    epochs = cfg["model"].get("epochs", 20)
    lr = cfg["model"]["learning_rate"]
    hidden_dim = cfg["model"]["hidden_dim"]
    num_layers = cfg["model"]["num_layers"]
    dropout = cfg["model"]["dropout"]
    num_stages = cfg["model"]["num_stages"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("--- [ChronoGuard] Phase 3: LSTM Forecaster Training ---")
    print(f"Loading raw datasets from {raw_dir}...")
    train_df = load_raw_dataset(raw_dir, split="train", sample_per_file=sample_per_file, selected_features=selected_features)
    test_df = load_raw_dataset(raw_dir, split="test", sample_per_file=sample_per_file, selected_features=selected_features)
    print(f"Loaded {len(train_df):,} train flows, {len(test_df):,} test flows.")

    # Windowing
    print("Generating windows and sequences...")
    train_windows = bucket_flows_into_windows(train_df, selected_features, window_size_flows)
    test_windows = bucket_flows_into_windows(test_df, selected_features, window_size_flows)

    X_train, y_train_prob, y_train_stage, _ = create_sequences(train_windows, seq_w, forecast_k)
    X_test, y_test_prob, y_test_stage, _ = create_sequences(test_windows, seq_w, forecast_k)

    N_tr, W, F_dim = X_train.shape
    N_te = X_test.shape[0]
    print(f"Dataset tensors: Train {X_train.shape}, Test {X_test.shape}")

    # Scaler
    scaler_path = "models/scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
        
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, F_dim))
        joblib.dump(scaler, scaler_path)

    # Scale sequences
    for i in range(W):
        X_train[:, i, :] = scaler.transform(X_train[:, i, :])
        X_test[:, i, :] = scaler.transform(X_test[:, i, :])

    # Convert to PyTorch tensors
    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train_prob = torch.tensor(y_train_prob, dtype=torch.float32)
    t_y_train_stage = torch.tensor(y_train_stage, dtype=torch.long)

    t_X_test = torch.tensor(X_test, dtype=torch.float32)
    t_y_test_prob = torch.tensor(y_test_prob, dtype=torch.float32)
    t_y_test_stage = torch.tensor(y_test_stage, dtype=torch.long)

    train_dataset = TensorDataset(t_X_train, t_y_train_prob, t_y_train_stage)
    test_dataset = TensorDataset(t_X_test, t_y_test_prob, t_y_test_stage)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights for stage loss to address benign traffic imbalance
    stage_counts = np.bincount(y_train_stage, minlength=num_stages)
    total_samples = len(y_train_stage)
    class_weights = [total_samples / (num_stages * max(count, 1)) for count in stage_counts]
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Instantiate Model
    model = ChronoGuardLSTM(
        input_dim=F_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        forecast_horizon_k=forecast_k,
        num_stages=num_stages,
    ).to(device)

    criterion_bce = nn.BCELoss()
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": []}

    print(f"\nTraining ChronoGuard LSTM for {epochs} epochs on CPU...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_prob, batch_stage in train_loader:
            batch_x = batch_x.to(device)
            batch_prob = batch_prob.to(device)
            batch_stage = batch_stage.to(device)

            optimizer.zero_grad()
            prob_pred, stage_logits, _ = model(batch_x)

            loss_prob = criterion_bce(prob_pred, batch_prob)
            loss_stage = criterion_ce(stage_logits, batch_stage)
            loss = loss_prob + loss_stage

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(batch_x)

        epoch_train_loss = running_loss / len(train_dataset)

        # Validation Loss
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_prob, batch_stage in test_loader:
                batch_x = batch_x.to(device)
                batch_prob = batch_prob.to(device)
                batch_stage = batch_stage.to(device)

                prob_pred, stage_logits, _ = model(batch_x)
                loss_prob = criterion_bce(prob_pred, batch_prob)
                loss_stage = criterion_ce(stage_logits, batch_stage)
                val_loss = loss_prob + loss_stage
                val_running_loss += val_loss.item() * len(batch_x)

        epoch_val_loss = val_running_loss / len(test_dataset)
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # Plot and save learning curves
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="#22D3EE", lw=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="#F59E0B", lw=2, linestyle="--")
    plt.title("ChronoGuard LSTM Training Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Multi-Task Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
    print("Saved results/training_curves.png")

    # Final Evaluation
    print("\nEvaluating LSTM forecaster on test holdout...")
    metrics = evaluate_lstm(model, test_loader, device, forecast_k=forecast_k)

    # Save artifacts
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": F_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "forecast_horizon_k": forecast_k,
        "num_stages": num_stages,
    }, "models/chronoguard_lstm.pt")
    print("Saved models/chronoguard_lstm.pt")

    with open("results/lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results/lstm_metrics.json")

    print("\n--- ChronoGuard LSTM Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"FPR:       {metrics['false_positive_rate']:.4f}")
    print("K-step Forecast Accuracies:")
    for h, m in metrics["k_step_forecast"].items():
        print(f"  {h}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}")

    return metrics


if __name__ == "__main__":
    train_lstm()
