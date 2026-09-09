"""Baseline classifier training and evaluation.

Implements scikit-learn Logistic Regression baseline trained on flattened
window features per AI-INSTRUCTIONS/01-ai-agent-instructions.md Phase 2
and AI-INSTRUCTIONS/05-ml-algorithms.md Section 5.

Saves:
- models/baseline_lr.pkl
- models/scaler.pkl
- results/baseline_metrics.json
"""

import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.data.load_raw import load_raw_dataset
from src.data.windowing import (
    bucket_flows_into_windows,
    create_sequences,
    get_feature_names,
)
from src.data.label_mapping import STAGE_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive security evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Binary FPR: Benign (class 0) vs Any Attack (classes 1..5)
    binary_true = (y_true > 0).astype(int)
    binary_pred = (y_pred > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Per-stage breakdown
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(STAGE_NAMES))),
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": "Logistic Regression Baseline",
        "accuracy": round(acc, 4),
        "precision": round(rec, 4),    # macro
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "total_test_samples": int(len(y_true)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
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


def train_baseline(
    config_path: str = "config.yaml",
    sample_per_file: int = 5000,
) -> Dict[str, Any]:
    """Train and evaluate baseline Logistic Regression on temporal split."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["dataset"]["raw_dir"]
    selected_features = cfg["dataset"]["selected_features"]
    window_size_flows = cfg["windowing"]["window_size_flows"]
    seq_w = cfg["windowing"]["sequence_length_w"]
    k_horizon = cfg["windowing"]["forecast_horizon_k"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("--- [ChronoGuard] Phase 2: Baseline Model Training ---")
    print("Loading training data (Mon-Wed split)...")
    train_df = load_raw_dataset(
        raw_dir=raw_dir,
        split="train",
        sample_per_file=sample_per_file,
        selected_features=selected_features,
    )
    print(f"Loaded {len(train_df):,} training flows.")

    print("Loading test data (Thu-Fri time-based holdout split)...")
    test_df = load_raw_dataset(
        raw_dir=raw_dir,
        split="test",
        sample_per_file=sample_per_file,
        selected_features=selected_features,
    )
    print(f"Loaded {len(test_df):,} test flows.")

    # Windowing
    print("Bucketing flows into windows...")
    train_windows = bucket_flows_into_windows(train_df, selected_features, window_size_flows)
    test_windows = bucket_flows_into_windows(test_df, selected_features, window_size_flows)
    print(f"Created {len(train_windows):,} train windows, {len(test_windows):,} test windows.")

    # Create sequence tensors
    X_train_seq, _, y_train, _ = create_sequences(train_windows, seq_w, k_horizon)
    X_test_seq, _, y_test, _ = create_sequences(test_windows, seq_w, k_horizon)

    # Flatten (N, W, feat) to (N, W*feat) for logistic regression
    N_tr, W, F_dim = X_train_seq.shape
    N_te = X_test_seq.shape[0]

    X_train_flat = X_train_seq.reshape(N_tr, W * F_dim)
    X_test_flat = X_test_seq.reshape(N_te, W * F_dim)

    # Fit Scaler on training window features
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    # Fit on per-window features (N_tr * W, F_dim)
    all_tr_windows = X_train_seq.reshape(-1, F_dim)
    scaler.fit(all_tr_windows)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Saved models/scaler.pkl")

    # Transform flattened sequences
    X_train_scaled = np.zeros_like(X_train_flat)
    X_test_scaled = np.zeros_like(X_test_flat)
    for i in range(W):
        X_train_scaled[:, i*F_dim:(i+1)*F_dim] = scaler.transform(X_train_seq[:, i, :])
        X_test_scaled[:, i*F_dim:(i+1)*F_dim] = scaler.transform(X_test_seq[:, i, :])

    # Train Logistic Regression
    print("Fitting Logistic Regression (multi-class, balanced)...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(X_train_scaled, y_train)

    # Evaluate
    print("Evaluating baseline on test split...")
    y_pred = lr.predict(X_test_scaled)
    metrics = compute_metrics(y_test, y_pred)

    # Save artifacts
    joblib.dump(lr, "models/baseline_lr.pkl")
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Baseline Results Summary ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"FPR:       {metrics['false_positive_rate']:.4f}")
    print("Saved models/baseline_lr.pkl and results/baseline_metrics.json\n")

    return metrics


if __name__ == "__main__":
    train_baseline()
