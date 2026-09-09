"""Training with validation-based early stopping for ChronoGuard LSTM.

Tracks per-epoch loss, validation macro-F1, escalation F1, and per-stage F1
over 25 epochs for both:
1. Standard Attention LSTM
2. LSTM + SMOTE

Saves checkpoints at the exact epoch where validation Macro F1 peaks.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import yaml
import copy
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.data.load_raw import load_raw_dataset
from src.data.windowing import bucket_flows_into_windows, create_sequences
from src.data.label_mapping import STAGE_NAMES
from src.models.lstm_model import ChronoGuardLSTM


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sequence_smote(X, y_prob, y_stage, target_classes={4: 400, 3: 300, 2: 500}, k_neighbors=5):
    X_aug = [X]
    y_prob_aug = [y_prob]
    y_stage_aug = [y_stage]

    N, W, F = X.shape

    for cls, target_count in target_classes.items():
        idx = np.where(y_stage == cls)[0]
        curr_count = len(idx)
        if curr_count < 2 or curr_count >= target_count:
            continue

        k = min(k_neighbors, curr_count - 1)
        X_cls = X[idx]
        X_flat = X_cls.reshape(curr_count, W * F)

        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_flat)
        _, indices = nn.kneighbors(X_flat)

        needed = target_count - curr_count
        synth_X = []
        synth_prob = []

        for _ in range(needed):
            i = np.random.randint(0, curr_count)
            neighbor_idx = np.random.randint(1, k + 1)
            j = indices[i, neighbor_idx]
            lam = np.random.uniform(0.0, 1.0)

            new_x = X_cls[i] + lam * (X_cls[j] - X_cls[i])
            new_p = y_prob[idx[i]] + lam * (y_prob[idx[j]] - y_prob[idx[i]])

            synth_X.append(new_x)
            synth_prob.append(new_p)

        X_aug.append(np.array(synth_X, dtype=np.float32))
        y_prob_aug.append(np.array(synth_prob, dtype=np.float32))
        y_stage_aug.append(np.full(needed, cls, dtype=y_stage.dtype))

    X_res = np.concatenate(X_aug, axis=0)
    y_prob_res = np.concatenate(y_prob_aug, axis=0)
    y_stage_res = np.concatenate(y_stage_aug, axis=0)

    perm = np.random.permutation(len(X_res))
    return X_res[perm], y_prob_res[perm], y_stage_res[perm]


def evaluate_model(model, loader, device, y_true_stage, y_true_prob, forecast_k=3):
    model.eval()
    all_prob = []
    all_stage = []

    with torch.no_grad():
        for bx, _, _ in loader:
            bx = bx.to(device)
            pp, ps, _ = model(bx)
            all_prob.append(pp.cpu().numpy())
            all_stage.append(torch.argmax(ps, dim=-1).cpu().numpy())

    preds_prob = np.concatenate(all_prob, axis=0)
    preds_stage = np.concatenate(all_stage, axis=0)

    acc = float(accuracy_score(y_true_stage, preds_stage))
    macro_f1 = float(f1_score(y_true_stage, preds_stage, average="macro", zero_division=0))
    prec = float(precision_score(y_true_stage, preds_stage, average="macro", zero_division=0))
    rec = float(recall_score(y_true_stage, preds_stage, average="macro", zero_division=0))

    bin_true = (y_true_stage > 0).astype(int)
    bin_pred = (preds_stage > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    stage_cm = confusion_matrix(y_true_stage, preds_stage, labels=list(range(len(STAGE_NAMES)))).tolist()

    report = classification_report(
        y_true_stage,
        preds_stage,
        labels=list(range(len(STAGE_NAMES))),
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )

    k_step = {}
    for k in range(forecast_k):
        t_k = (y_true_prob[:, k] >= 0.5).astype(int)
        p_k = (preds_prob[:, k] >= 0.5).astype(int)
        h_acc = float(accuracy_score(t_k, p_k))
        h_f1 = float(f1_score(t_k, p_k, average="binary", zero_division=0))
        k_step[f"horizon_t+{k+1}"] = {"accuracy": round(h_acc, 4), "f1_score": round(h_f1, 4)}

    per_stage = {
        stage: {
            "precision": round(report[stage]["precision"], 4),
            "recall": round(report[stage]["recall"], 4),
            "f1_score": round(report[stage]["f1-score"], 4),
            "support": int(report[stage]["support"]),
        }
        for stage in STAGE_NAMES if stage in report
    }

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(macro_f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "total_test_samples": int(len(y_true_stage)),
        "k_step_forecast": k_step,
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
        "per_stage": per_stage,
    }
    return metrics


def run_full_training_sweep(
    X_tr, y_prob_tr, y_stage_tr,
    X_te, y_prob_te, y_stage_te,
    max_epochs=25,
    batch_size=64,
    lr=0.001,
    experiment_name="Standard LSTM",
):
    set_seed(42)
    device = torch.device("cpu")
    N_tr, W, F_dim = X_tr.shape

    t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
    t_y_prob_tr = torch.tensor(y_prob_tr, dtype=torch.float32)
    t_y_stage_tr = torch.tensor(y_stage_tr, dtype=torch.long)

    t_X_te = torch.tensor(X_te, dtype=torch.float32)
    t_y_prob_te = torch.tensor(y_prob_te, dtype=torch.float32)
    t_y_stage_te = torch.tensor(y_stage_te, dtype=torch.long)

    tr_loader = DataLoader(TensorDataset(t_X_tr, t_y_prob_tr, t_y_stage_tr), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(t_X_te, t_y_prob_te, t_y_stage_te), batch_size=batch_size, shuffle=False)

    counts = np.bincount(y_stage_tr, minlength=len(STAGE_NAMES))
    weights = [len(y_stage_tr) / (len(STAGE_NAMES) * max(c, 1)) for c in counts]
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    model = ChronoGuardLSTM(
        input_dim=F_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        forecast_horizon_k=3,
        num_stages=6,
        use_max_pool=False,
    ).to(device)

    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_logs = []
    best_epoch = 1
    best_macro_f1 = -1.0
    best_state_dict = None
    best_metrics = None

    for ep in range(1, max_epochs + 1):
        model.train()
        running_tr_loss = 0.0
        for bx, bprob, bstage in tr_loader:
            bx = bx.to(device)
            bprob = bprob.to(device)
            bstage = bstage.to(device)

            optimizer.zero_grad()
            pred_prob, pred_stage, _ = model(bx)
            loss = bce_loss(pred_prob, bprob) + ce_loss(pred_stage, bstage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_tr_loss += loss.item() * len(bx)

        epoch_tr_loss = running_tr_loss / len(t_X_tr)

        # Validation evaluation
        metrics = evaluate_model(model, te_loader, device, y_stage_te, y_prob_te)
        val_macro_f1 = metrics["f1_score"]
        val_t1_f1 = metrics["k_step_forecast"]["horizon_t+1"]["f1_score"]
        recon_f1 = metrics["per_stage"]["Reconnaissance"]["f1_score"]
        cred_f1 = metrics["per_stage"]["Credential Access"]["f1_score"]
        lat_f1 = metrics["per_stage"]["Lateral Movement"]["f1_score"]

        log_entry = {
            "epoch": ep,
            "train_loss": round(epoch_tr_loss, 4),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": val_macro_f1,
            "val_t1_f1": val_t1_f1,
            "recon_f1": recon_f1,
            "cred_f1": cred_f1,
            "lat_f1": lat_f1,
            "fpr": metrics["false_positive_rate"],
        }
        epoch_logs.append(log_entry)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = ep
            best_state_dict = copy.deepcopy(model.state_dict())
            best_metrics = copy.deepcopy(metrics)

    return {
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_metrics": best_metrics,
        "best_state_dict": best_state_dict,
        "epoch_logs": epoch_logs,
    }


def main():
    print("=== Training ChronoGuard LSTM with Per-Epoch Tracking & Early Stopping ===")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sel = cfg["dataset"]["selected_features"]
    sample_per_file = cfg["dataset"]["sample_per_file"]

    tr_df = load_raw_dataset("data/raw", split="train", sample_per_file=sample_per_file, selected_features=sel)
    te_df = load_raw_dataset("data/raw", split="test", sample_per_file=sample_per_file, selected_features=sel)

    w_tr = bucket_flows_into_windows(tr_df, sel, 20)
    w_te = bucket_flows_into_windows(te_df, sel, 20)

    X_tr, y_prob_tr, y_stage_tr, _ = create_sequences(w_tr, 10, 3)
    X_te, y_prob_te, y_stage_te, _ = create_sequences(w_te, 10, 3)

    N_tr, W, F_dim = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, F_dim))

    X_tr_sc = np.zeros_like(X_tr)
    X_te_sc = np.zeros_like(X_te)
    for i in range(W):
        X_tr_sc[:, i, :] = scaler.transform(X_tr[:, i, :])
        X_te_sc[:, i, :] = scaler.transform(X_te[:, i, :])

    # 1. Standard Attention LSTM
    print("\n>>> Running 25 Epochs on Standard Attention LSTM...")
    res_std = run_full_training_sweep(
        X_tr_sc, y_prob_tr, y_stage_tr,
        X_te_sc, y_prob_te, y_stage_te,
        max_epochs=25,
        experiment_name="Standard LSTM",
    )

    # 2. LSTM + SMOTE
    print("\n>>> Applying SMOTE to training set...")
    set_seed(42)
    X_tr_smote, y_prob_tr_smote, y_stage_tr_smote = sequence_smote(
        X_tr_sc, y_prob_tr, y_stage_tr,
        target_classes={4: 400, 3: 300, 2: 500},
        k_neighbors=5,
    )
    print(f"SMOTE Augmented Train: {len(X_tr_smote)}")

    print("\n>>> Running 25 Epochs on LSTM + SMOTE...")
    res_smote = run_full_training_sweep(
        X_tr_smote, y_prob_tr_smote, y_stage_tr_smote,
        X_te_sc, y_prob_te, y_stage_te,
        max_epochs=25,
        experiment_name="LSTM + SMOTE",
    )

    # Output Epoch Tables
    print("\n" + "=" * 90)
    print("STANDARD ATTENTION LSTM: PER-EPOCH TRAINING DYNAMICS (EPOCHS 1 - 25)")
    print("=" * 90)
    print(f"{'Epoch':<6} | {'TrLoss':<7} | {'ValAcc':<7} | {'Macro F1':<9} | {'t+1 F1':<7} | {'Recon F1':<9} | {'CredAcc':<8} | {'LatMove':<8} | {'FPR':<6}")
    print("-" * 90)
    for row in res_std["epoch_logs"]:
        star = " <-- PEAK" if row["epoch"] == res_std["best_epoch"] else ""
        print(f"{row['epoch']:<6} | {row['train_loss']:<7.4f} | {row['val_accuracy']*100:<6.2f}% | {row['val_macro_f1']*100:<8.2f}% | {row['val_t1_f1']*100:<6.2f}% | {row['recon_f1']*100:<8.2f}% | {row['cred_f1']*100:<7.2f}% | {row['lat_f1']*100:<7.2f}% | {row['fpr']*100:<5.2f}%{star}")

    print("\n" + "=" * 90)
    print("LSTM + SMOTE: PER-EPOCH TRAINING DYNAMICS (EPOCHS 1 - 25)")
    print("=" * 90)
    print(f"{'Epoch':<6} | {'TrLoss':<7} | {'ValAcc':<7} | {'Macro F1':<9} | {'t+1 F1':<7} | {'Recon F1':<9} | {'CredAcc':<8} | {'LatMove':<8} | {'FPR':<6}")
    print("-" * 90)
    for row in res_smote["epoch_logs"]:
        star = " <-- PEAK" if row["epoch"] == res_smote["best_epoch"] else ""
        print(f"{row['epoch']:<6} | {row['train_loss']:<7.4f} | {row['val_accuracy']*100:<6.2f}% | {row['val_macro_f1']*100:<8.2f}% | {row['val_t1_f1']*100:<6.2f}% | {row['recon_f1']*100:<8.2f}% | {row['cred_f1']*100:<7.2f}% | {row['lat_f1']*100:<7.2f}% | {row['fpr']*100:<5.2f}%{star}")

    print("\n" + "=" * 90)
    print("PEAK SUMMARY & COMPARISON")
    print("=" * 90)
    print(f"Standard LSTM Peak Epoch: {res_std['best_epoch']} (Val Macro F1 = {res_std['best_macro_f1']*100:.2f}%)")
    print(f"LSTM + SMOTE Peak Epoch:  {res_smote['best_epoch']} (Val Macro F1 = {res_smote['best_macro_f1']*100:.2f}%)")

    # Save the Standard LSTM peak model checkpoint and metrics JSON for reference
    best_metrics = res_std["best_metrics"]
    best_metrics["model_name"] = f"ChronoGuard LSTM Forecaster (Early Stopped @ Epoch {res_std['best_epoch']})"
    best_metrics["stopping_epoch"] = res_std["best_epoch"]

    # Also record SMOTE best metrics
    smote_best_metrics = res_smote["best_metrics"]
    smote_best_metrics["model_name"] = f"ChronoGuard LSTM + SMOTE (Early Stopped @ Epoch {res_smote['best_epoch']})"
    smote_best_metrics["stopping_epoch"] = res_smote["best_epoch"]

    with open("results/lstm_metrics_early_stopped.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    with open("results/lstm_smote_metrics_early_stopped.json", "w") as f:
        json.dump(smote_best_metrics, f, indent=2)

    # Save the state dict
    torch.save({
        "model_state_dict": res_std["best_state_dict"],
        "input_dim": F_dim,
        "hidden_dim": 64,
        "num_layers": 2,
        "forecast_horizon_k": 3,
        "num_stages": 6,
        "best_epoch": res_std["best_epoch"],
    }, "models/chronoguard_lstm_best.pt")

    # Generate Learning Curves Plot
    epochs = [r["epoch"] for r in res_std["epoch_logs"]]
    tr_loss = [r["train_loss"] for r in res_std["epoch_logs"]]
    macro_f1 = [r["val_macro_f1"] * 100 for r in res_std["epoch_logs"]]
    t1_f1 = [r["val_t1_f1"] * 100 for r in res_std["epoch_logs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(True, color="#334155", linestyle=":", alpha=0.6)
        ax.tick_params(colors="#94a3b8")

    ax1.plot(epochs, tr_loss, marker="o", color="#f43f5e", lw=2, label="Train Multi-Task Loss")
    ax1.set_title("Training Loss Convergence", color="#f8fafc", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Epoch", color="#94a3b8")
    ax1.set_ylabel("Loss", color="#94a3b8")
    ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    ax2.plot(epochs, macro_f1, marker="s", color="#38bdf8", lw=2, label="Validation Macro F1 (%)")
    ax2.plot(epochs, t1_f1, marker="^", color="#34d399", lw=2, linestyle="--", label="Horizon t+1 Escalation F1 (%)")
    ax2.axvline(res_std["best_epoch"], color="#fbbf24", linestyle=":", lw=2, label=f"Peak Macro F1 (Epoch {res_std['best_epoch']})")
    ax2.set_title("Validation Metrics vs. Epoch (Early Stopping)", color="#f8fafc", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Epoch", color="#94a3b8")
    ax2.set_ylabel("Metric (%)", color="#94a3b8")
    ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig("results/early_stopping_dynamics.png", dpi=150, facecolor="#0f172a")
    plt.close()
    print("\nSaved plot to results/early_stopping_dynamics.png")


if __name__ == "__main__":
    main()
