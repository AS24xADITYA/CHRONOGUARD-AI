"""Comprehensive experimental evaluation for ChronoGuard:
1. Baseline (Standard)
2. Baseline + SMOTE (training set only)
3. LSTM (Standard Attention)
4. LSTM + SMOTE (training set only)
5. LSTM + Max-Pooling Branch (Attention + Max-Pool)
6. LSTM + Max-Pooling Branch + SMOTE

Evaluates on the exact same held-out test split (N=1,805 sequences)
under the per-file chronological 75/25 split.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import copy
import joblib
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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


def sequence_smote(X, y_prob, y_stage, target_classes={4: 400, 3: 400, 2: 500}, k_neighbors=5):
    """Synthetic Minority Over-sampling Technique adapted for sequence tensors (N, W, F).
    Operates strictly on the training set.
    """
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
        X_cls = X[idx] # (curr_count, W, F)
        X_flat = X_cls.reshape(curr_count, W * F)

        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_flat)
        _, indices = nn.kneighbors(X_flat)

        needed = target_count - curr_count
        synth_X = []
        synth_prob = []

        for _ in range(needed):
            i = np.random.randint(0, curr_count)
            # Pick a neighbor among k nearest (excluding self at 0)
            neighbor_idx = np.random.randint(1, k + 1)
            j = indices[i, neighbor_idx]
            lam = np.random.uniform(0.0, 1.0)

            # Linear interpolation in continuous feature space
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

    # Shuffle
    perm = np.random.permutation(len(X_res))
    return X_res[perm], y_prob_res[perm], y_stage_res[perm]


def evaluate_stage_and_horizon(y_true_stage, y_pred_stage, y_true_prob, y_pred_prob):
    acc = float(accuracy_score(y_true_stage, y_pred_stage))
    prec = float(precision_score(y_true_stage, y_pred_stage, average="macro", zero_division=0))
    rec = float(recall_score(y_true_stage, y_pred_stage, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_stage, y_pred_stage, average="macro", zero_division=0))

    # Binary FPR
    bin_true = (y_true_stage > 0).astype(int)
    bin_pred = (y_pred_stage > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    report = classification_report(
        y_true_stage,
        y_pred_stage,
        labels=list(range(len(STAGE_NAMES))),
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )

    horizons = {}
    if y_pred_prob is not None:
        for k in range(3):
            t_k = (y_true_prob[:, k] >= 0.5).astype(int)
            p_k = (y_pred_prob[:, k] >= 0.5).astype(int)
            h_acc = float(accuracy_score(t_k, p_k))
            h_f1 = float(f1_score(t_k, p_k, average="binary", zero_division=0))
            horizons[f"t+{k+1}"] = {"acc": round(h_acc, 4), "f1": round(h_f1, 4)}

    return {
        "acc": round(acc, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "report": report,
        "horizons": horizons,
    }


def train_and_eval_lstm(
    X_tr, y_prob_tr, y_stage_tr,
    X_te, y_prob_te, y_stage_te,
    use_max_pool=False,
    epochs=15,
    batch_size=64,
    lr=0.001,
    name="LSTM",
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
        use_max_pool=use_max_pool,
    ).to(device)

    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        for bx, bprob, bstage in tr_loader:
            optimizer.zero_grad()
            pred_prob, pred_stage, _ = model(bx)
            loss = bce_loss(pred_prob, bprob) + ce_loss(pred_stage, bstage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Eval
    model.eval()
    all_prob = []
    all_stage = []
    with torch.no_grad():
        for bx, _, _ in te_loader:
            pp, ps, _ = model(bx)
            all_prob.append(pp.numpy())
            all_stage.append(torch.argmax(ps, dim=-1).numpy())

    preds_prob = np.concatenate(all_prob, axis=0)
    preds_stage = np.concatenate(all_stage, axis=0)

    res = evaluate_stage_and_horizon(y_stage_te, preds_stage, y_prob_te, preds_prob)
    return res


def main():
    print("=== ChronoGuard Architectural & Data Experiments ===")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sel = cfg["dataset"]["selected_features"]
    sample_per_file = cfg["dataset"]["sample_per_file"]

    print(f"Loading raw data (sample_per_file = {sample_per_file})...")
    tr_df = load_raw_dataset("data/raw", split="train", sample_per_file=sample_per_file, selected_features=sel)
    te_df = load_raw_dataset("data/raw", split="test", sample_per_file=sample_per_file, selected_features=sel)

    print("Windowing...")
    w_tr = bucket_flows_into_windows(tr_df, sel, 20)
    w_te = bucket_flows_into_windows(te_df, sel, 20)

    X_tr, y_prob_tr, y_stage_tr, _ = create_sequences(w_tr, 10, 3)
    X_te, y_prob_te, y_stage_te, _ = create_sequences(w_te, 10, 3)

    print(f"Base Sequences: Train={len(X_tr)}, Test={len(X_te)}")
    print("Training Stage Distribution:")
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s:<20}: {np.sum(y_stage_tr == i)}")

    # Standard Scaling
    N_tr, W, F_dim = X_tr.shape
    N_te = X_te.shape[0]

    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, F_dim))

    X_tr_sc = np.zeros_like(X_tr)
    X_te_sc = np.zeros_like(X_te)
    for i in range(W):
        X_tr_sc[:, i, :] = scaler.transform(X_tr[:, i, :])
        X_te_sc[:, i, :] = scaler.transform(X_te[:, i, :])

    # SMOTE oversampled training set
    print("\nApplying SMOTE to minority classes in training set (Lateral Movement, Initial Access, Credential Access)...")
    set_seed(42)
    X_tr_smote, y_prob_tr_smote, y_stage_tr_smote = sequence_smote(
        X_tr_sc, y_prob_tr, y_stage_tr,
        target_classes={4: 400, 3: 300, 2: 500},
        k_neighbors=5
    )
    print(f"SMOTE Augmented Train Sequences: {len(X_tr_smote)}")
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s:<20}: {np.sum(y_stage_tr_smote == i)}")

    results = {}

    # 1. Baseline Standard
    print("\n--- 1. Evaluating Baseline (Standard) ---")
    clf_base = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf_base.fit(X_tr_sc.reshape(N_tr, W * F_dim), y_stage_tr)
    p_bl = clf_base.predict(X_te_sc.reshape(N_te, W * F_dim))
    results["Baseline (Standard)"] = evaluate_stage_and_horizon(y_stage_te, p_bl, y_prob_te, None)

    # 2. Baseline + SMOTE
    print("--- 2. Evaluating Baseline + SMOTE ---")
    clf_smote = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf_smote.fit(X_tr_smote.reshape(len(X_tr_smote), W * F_dim), y_stage_tr_smote)
    p_bl_smote = clf_smote.predict(X_te_sc.reshape(N_te, W * F_dim))
    results["Baseline + SMOTE"] = evaluate_stage_and_horizon(y_stage_te, p_bl_smote, y_prob_te, None)

    # 3. LSTM Standard (Attention only)
    print("--- 3. Evaluating LSTM Standard (Attention only) ---")
    results["LSTM Standard"] = train_and_eval_lstm(
        X_tr_sc, y_prob_tr, y_stage_tr,
        X_te_sc, y_prob_te, y_stage_te,
        use_max_pool=False, epochs=15, name="LSTM Standard"
    )

    # 4. LSTM + SMOTE
    print("--- 4. Evaluating LSTM + SMOTE ---")
    results["LSTM + SMOTE"] = train_and_eval_lstm(
        X_tr_smote, y_prob_tr_smote, y_stage_tr_smote,
        X_te_sc, y_prob_te, y_stage_te,
        use_max_pool=False, epochs=15, name="LSTM + SMOTE"
    )

    # 5. LSTM + Max-Pooling Branch
    print("--- 5. Evaluating LSTM + Max-Pooling Branch ---")
    results["LSTM + Max-Pooling"] = train_and_eval_lstm(
        X_tr_sc, y_prob_tr, y_stage_tr,
        X_te_sc, y_prob_te, y_stage_te,
        use_max_pool=True, epochs=15, name="LSTM + Max-Pooling"
    )

    # 6. LSTM + Max-Pooling + SMOTE
    print("--- 6. Evaluating LSTM + Max-Pooling + SMOTE ---")
    results["LSTM + Max-Pool + SMOTE"] = train_and_eval_lstm(
        X_tr_smote, y_prob_tr_smote, y_stage_tr_smote,
        X_te_sc, y_prob_te, y_stage_te,
        use_max_pool=True, epochs=15, name="LSTM + Max-Pool + SMOTE"
    )

    # Print Summary Table
    print("\n\n========================================================")
    print("        CONSOLIDATED EXPERIMENTAL COMPARISON TABLE      ")
    print("========================================================")
    header = f"{'Model Variant':<26} | {'Acc':>6} | {'Macro F1':>8} | {'FPR':>6} | {'t+1 F1':>6} | {'Recon F1':>8} | {'CredAcc':>8} | {'LatMove':>8}"
    print(header)
    print("-" * len(header))

    for name, r in results.items():
        rep = r["report"]
        t1_f1 = f"{r['horizons']['t+1']['f1']*100:.1f}%" if "t+1" in r["horizons"] else "N/A"
        recon = f"{rep.get('Reconnaissance', {}).get('f1-score', 0)*100:.1f}%"
        cred = f"{rep.get('Credential Access', {}).get('f1-score', 0)*100:.1f}%"
        lat = f"{rep.get('Lateral Movement', {}).get('f1-score', 0)*100:.1f}%"

        print(f"{name:<26} | {r['acc']*100:>5.1f}% | {r['f1']*100:>7.1f}% | {r['fpr']*100:>5.1f}% | {t1_f1:>6} | {recon:>8} | {cred:>8} | {lat:>8}")

    print("========================================================\n")


if __name__ == "__main__":
    main()
