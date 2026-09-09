"""5-Seed Benchmark using RobustScaler (Full Train + 5-sigma clip).

Tests whether switching from StandardScaler to RobustScaler:
1. Fixes Baseline Reconnaissance and Credential Access collapse.
2. Reduces seed-to-seed instability in the ChronoGuard LSTM.
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
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loader(model, loader, device, y_true_stage, y_true_prob, forecast_k=3):
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

    report = classification_report(
        y_true_stage,
        preds_stage,
        labels=list(range(len(STAGE_NAMES))),
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )

    t1_true = (y_true_prob[:, 0] >= 0.5).astype(int)
    t1_pred = (preds_prob[:, 0] >= 0.5).astype(int)
    t1_f1 = float(f1_score(t1_true, t1_pred, average="binary", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "t1_f1": t1_f1,
        "per_stage": {
            s: report[s]["f1-score"] for s in STAGE_NAMES if s in report
        },
    }


def evaluate_baseline(clf, X_scaled, y_true_stage):
    N, W, F = X_scaled.shape
    preds = clf.predict(X_scaled.reshape(N, W * F))
    acc = float(accuracy_score(y_true_stage, preds))
    macro_f1 = float(f1_score(y_true_stage, preds, average="macro", zero_division=0))
    bin_true = (y_true_stage > 0).astype(int)
    bin_pred = (preds > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    report = classification_report(
        y_true_stage,
        preds,
        labels=list(range(len(STAGE_NAMES))),
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "fpr": fpr,
        "per_stage": {s: report[s]["f1-score"] for s in STAGE_NAMES if s in report},
    }


def main():
    print("=== ChronoGuard 5-Seed Stability Test: RobustScaler (Full Train + 5-sigma clip) ===")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sel = cfg["dataset"]["selected_features"]
    sample_per_file = cfg["dataset"]["sample_per_file"]

    # Load 3-way chronological partitions
    tr_df = load_raw_dataset("data/raw", split="train", sample_per_file=sample_per_file, selected_features=sel)
    val_df = load_raw_dataset("data/raw", split="val", sample_per_file=sample_per_file, selected_features=sel)
    te_df = load_raw_dataset("data/raw", split="test", sample_per_file=sample_per_file, selected_features=sel)

    w_tr = bucket_flows_into_windows(tr_df, sel, 20)
    w_val = bucket_flows_into_windows(val_df, sel, 20)
    w_te = bucket_flows_into_windows(te_df, sel, 20)

    X_tr, y_prob_tr, y_stage_tr, _ = create_sequences(w_tr, 10, 3)
    X_val, y_prob_val, y_stage_val, _ = create_sequences(w_val, 10, 3)
    X_te, y_prob_te, y_stage_te, _ = create_sequences(w_te, 10, 3)

    N_tr, W, F_dim = X_tr.shape

    # Fit RobustScaler on FULL training set
    print("Fitting RobustScaler on FULL training set (all classes) with 5-sigma clipping...")
    scaler = RobustScaler()
    scaler.fit(X_tr.reshape(-1, F_dim))

    def scale_and_clip(X):
        X_sc = np.zeros_like(X)
        for i in range(W):
            sc = scaler.transform(X[:, i, :])
            sc = np.clip(sc, -5.0, 5.0)
            X_sc[:, i, :] = sc
        return X_sc

    X_tr_sc = scale_and_clip(X_tr)
    X_val_sc = scale_and_clip(X_val)
    X_te_sc = scale_and_clip(X_te)

    # Tensors for Val and Test (fixed)
    t_X_val = torch.tensor(X_val_sc, dtype=torch.float32)
    t_y_prob_val = torch.tensor(y_prob_val, dtype=torch.float32)
    t_y_stage_val = torch.tensor(y_stage_val, dtype=torch.long)

    t_X_te = torch.tensor(X_te_sc, dtype=torch.float32)
    t_y_prob_te = torch.tensor(y_prob_te, dtype=torch.float32)
    t_y_stage_te = torch.tensor(y_stage_te, dtype=torch.long)

    val_loader = DataLoader(TensorDataset(t_X_val, t_y_prob_val, t_y_stage_val), batch_size=64, shuffle=False)
    te_loader = DataLoader(TensorDataset(t_X_te, t_y_prob_te, t_y_stage_te), batch_size=64, shuffle=False)

    counts = np.bincount(y_stage_tr, minlength=len(STAGE_NAMES))
    weights = [len(y_stage_tr) / (len(STAGE_NAMES) * max(c, 1)) for c in counts]
    weights_t = torch.tensor(weights, dtype=torch.float32)

    device = torch.device("cpu")
    seeds = [42, 123, 456, 789, 2024]
    lstm_results = []
    baseline_results = []

    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n[Run {run_idx}/5] Random Seed = {seed}")
        set_seed(seed)

        # Baseline run
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        clf.fit(X_tr_sc.reshape(N_tr, W * F_dim), y_stage_tr)
        bl_eval = evaluate_baseline(clf, X_te_sc, y_stage_te)
        bl_eval["seed"] = seed
        baseline_results.append(bl_eval)
        print(f"  Baseline: Test Macro F1 = {bl_eval['macro_f1']*100:.2f}%, Recon F1 = {bl_eval['per_stage']['Reconnaissance']*100:.2f}%, Cred F1 = {bl_eval['per_stage']['Credential Access']*100:.2f}%")

        # LSTM run
        t_X_tr = torch.tensor(X_tr_sc, dtype=torch.float32)
        t_y_prob_tr = torch.tensor(y_prob_tr, dtype=torch.float32)
        t_y_stage_tr = torch.tensor(y_stage_tr, dtype=torch.long)
        tr_loader = DataLoader(TensorDataset(t_X_tr, t_y_prob_tr, t_y_stage_tr), batch_size=64, shuffle=True)

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
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val_f1 = -1.0
        best_val_ep = 1
        best_weights = None

        for ep in range(1, 16):
            model.train()
            for bx, bprob, bstage in tr_loader:
                optimizer.zero_grad()
                pp, ps, _ = model(bx)
                l = bce_loss(pp, bprob) + ce_loss(ps, bstage)
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            val_res = evaluate_loader(model, val_loader, device, y_stage_val, y_prob_val)
            val_f1 = val_res["macro_f1"]

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_ep = ep
                best_weights = copy.deepcopy(model.state_dict())

        # Evaluate selected epoch checkpoint ONCE on untouched test set
        model.load_state_dict(best_weights)
        test_eval = evaluate_loader(model, te_loader, device, y_stage_te, y_prob_te)
        test_eval["seed"] = seed
        test_eval["selected_epoch"] = best_val_ep
        test_eval["val_macro_f1"] = best_val_f1
        lstm_results.append(test_eval)

        print(f"  LSTM: Peak Val Epoch = {best_val_ep} (Val F1 = {best_val_f1*100:.2f}%)")
        print(f"  LSTM: Test Macro F1 = {test_eval['macro_f1']*100:.2f}%, Recon F1 = {test_eval['per_stage']['Reconnaissance']*100:.2f}%, Cred F1 = {test_eval['per_stage']['Credential Access']*100:.2f}%, t+1 F1 = {test_eval['t1_f1']*100:.2f}%")

    # Output Full Multi-Seed Table
    print("\n\n" + "=" * 105)
    print("        ROBUSTSCALER (FULL TRAIN + 5s CLIP): 5-SEED STABILITY & REPRODUCIBILITY RESULTS")
    print("=" * 105)
    fmt = "{:<8} | {:<5} | {:>9} | {:>9} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}"
    print(fmt.format("Model", "Seed", "StopEpoch", "TestMacroF1", "Acc", "t+1 F1", "Recon F1", "CredAcc", "LatMove", "Impact"))
    print("-" * 105)

    for r in baseline_results:
        p = r["per_stage"]
        print(fmt.format(
            "Baseline",
            r["seed"],
            "N/A",
            f"{r['macro_f1']*100:.2f}%",
            f"{r['accuracy']*100:.1f}%",
            "N/A",
            f"{p['Reconnaissance']*100:.1f}%",
            f"{p['Credential Access']*100:.1f}%",
            f"{p['Lateral Movement']*100:.1f}%",
            f"{p['Impact']*100:.1f}%",
        ))

    print("-" * 105)
    for r in lstm_results:
        p = r["per_stage"]
        print(fmt.format(
            "LSTM",
            r["seed"],
            f"Ep {r['selected_epoch']}",
            f"{r['macro_f1']*100:.2f}%",
            f"{r['accuracy']*100:.1f}%",
            f"{r['t1_f1']*100:.1f}%",
            f"{p['Reconnaissance']*100:.1f}%",
            f"{p['Credential Access']*100:.1f}%",
            f"{p['Lateral Movement']*100:.1f}%",
            f"{p['Impact']*100:.1f}%",
        ))

    print("=" * 105)

    # Statistical Aggregations
    bl_f1s = [r["macro_f1"] * 100 for r in baseline_results]
    lstm_f1s = [r["macro_f1"] * 100 for r in lstm_results]
    lstm_t1s = [r["t1_f1"] * 100 for r in lstm_results]
    lstm_recons = [r["per_stage"]["Reconnaissance"] * 100 for r in lstm_results]
    lstm_creds = [r["per_stage"]["Credential Access"] * 100 for r in lstm_results]
    lstm_epochs = [r["selected_epoch"] for r in lstm_results]

    mode_epoch = max(set(lstm_epochs), key=lstm_epochs.count)
    print("\n--- Summary Statistics Across 5 Seeds (RobustScaler) ---")
    print(f"Baseline Test Macro F1:       Mean = {np.mean(bl_f1s):.2f}% ± {np.std(bl_f1s):.2f}%")
    print(f"LSTM Selected Stop Epoch:     Values = {lstm_epochs} (Mode = {mode_epoch})")
    print(f"LSTM Test Macro F1:           Mean = {np.mean(lstm_f1s):.2f}% ± {np.std(lstm_f1s):.2f}% (Range: {np.min(lstm_f1s):.2f}% - {np.max(lstm_f1s):.2f}%)")
    print(f"LSTM Escalation t+1 F1:       Mean = {np.mean(lstm_t1s):.2f}% ± {np.std(lstm_t1s):.2f}% (Range: {np.min(lstm_t1s):.2f}% - {np.max(lstm_t1s):.2f}%)")
    print(f"LSTM Reconnaissance F1:       Mean = {np.mean(lstm_recons):.2f}% ± {np.std(lstm_recons):.2f}% (Range: {np.min(lstm_recons):.2f}% - {np.max(lstm_recons):.2f}%)")
    print(f"LSTM Credential Access F1:    Mean = {np.mean(lstm_creds):.2f}% ± {np.std(lstm_creds):.2f}% (Range: {np.min(lstm_creds):.2f}% - {np.max(lstm_creds):.2f}%)")


if __name__ == "__main__":
    main()
