"""Rigorous 3-way chronological split training and validation for ChronoGuard.

Three completely disjoint chronological partitions per file:
- Train: First 65% of capture
- Validation: Next 10% of capture (used exclusively for early stopping selection)
- Test: Final 25% of capture (strictly untouched until final single evaluation)

Tracks per-epoch dynamics on the validation slice and evaluates the selected
checkpoint on the untouched test slice.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import yaml
import copy
import joblib
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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


def sequence_smote(X, y_prob, y_stage, target_classes={4: 300, 3: 200, 2: 400}, k_neighbors=5):
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


def evaluate_dataset(model, loader, device, y_true_stage, y_true_prob, forecast_k=3, model_name="ChronoGuard LSTM"):
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

    return {
        "model_name": model_name,
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


def evaluate_baseline(clf, X_scaled, y_true_stage, y_true_prob):
    N, W, F = X_scaled.shape
    X_flat = X_scaled.reshape(N, W * F)
    preds_stage = clf.predict(X_flat)

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

    per_stage = {
        stage: {
            "precision": round(report[stage]["precision"], 4),
            "recall": round(report[stage]["recall"], 4),
            "f1_score": round(report[stage]["f1-score"], 4),
            "support": int(report[stage]["support"]),
        }
        for stage in STAGE_NAMES if stage in report
    }

    return {
        "model_name": "Logistic Regression Baseline",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(macro_f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "total_test_samples": int(len(y_true_stage)),
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


def main():
    print("=== ChronoGuard Strict 3-Way Chronological Split & Evaluation ===")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sel = cfg["dataset"]["selected_features"]
    sample_per_file = cfg["dataset"]["sample_per_file"]

    print("Loading 3-way chronological partitions from data/raw...")
    tr_df = load_raw_dataset("data/raw", split="train", sample_per_file=sample_per_file, selected_features=sel)
    val_df = load_raw_dataset("data/raw", split="val", sample_per_file=sample_per_file, selected_features=sel)
    te_df = load_raw_dataset("data/raw", split="test", sample_per_file=sample_per_file, selected_features=sel)

    print(f"Loaded flows: Train={len(tr_df):,}, Val={len(val_df):,}, Test={len(te_df):,}")

    w_tr = bucket_flows_into_windows(tr_df, sel, 20)
    w_val = bucket_flows_into_windows(val_df, sel, 20)
    w_te = bucket_flows_into_windows(te_df, sel, 20)

    X_tr, y_prob_tr, y_stage_tr, _ = create_sequences(w_tr, 10, 3)
    X_val, y_prob_val, y_stage_val, _ = create_sequences(w_val, 10, 3)
    X_te, y_prob_te, y_stage_te, _ = create_sequences(w_te, 10, 3)

    print(f"\nSequence Dimensions: Train={X_tr.shape}, Val={X_val.shape}, Test={X_te.shape}")

    print("\nStage Class Counts Across Splits:")
    print(f"{'Stage':<20} | {'Train':>7} | {'Val':>6} | {'Test':>6}")
    print("-" * 46)
    for i, s in enumerate(STAGE_NAMES):
        c_tr = int(np.sum(y_stage_tr == i))
        c_val = int(np.sum(y_stage_val == i))
        c_te = int(np.sum(y_stage_te == i))
        print(f"{s:<20} | {c_tr:>7} | {c_val:>6} | {c_te:>6}")
    print("-" * 46)
    print(f"{'Total':<20} | {len(y_stage_tr):>7} | {len(y_stage_val):>6} | {len(y_stage_te):>6}")

    # Scaler fit STRICTLY on Train split
    N_tr, W, F_dim = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, F_dim))

    joblib.dump(scaler, "models/scaler.pkl")

    def scale_seq(X):
        X_sc = np.zeros_like(X)
        for i in range(W):
            X_sc[:, i, :] = scaler.transform(X[:, i, :])
        return X_sc

    X_tr_sc = scale_seq(X_tr)
    X_val_sc = scale_seq(X_val)
    X_te_sc = scale_seq(X_te)

    # 1. Baseline: Train on Train (65%), Evaluate ONCE on Untouched Test (25%)
    print("\n--- 1. Training Baseline on Train split (first 65%) ---")
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(X_tr_sc.reshape(N_tr, W * F_dim), y_stage_tr)
    joblib.dump(clf, "models/baseline_lr.pkl")

    base_metrics = evaluate_baseline(clf, X_te_sc, y_stage_te, y_prob_te)
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(base_metrics, f, indent=2)
    print("Baseline evaluated on untouched Test split:")
    print(f"  Accuracy: {base_metrics['accuracy']*100:.2f}%, Macro F1: {base_metrics['f1_score']*100:.2f}%, FPR: {base_metrics['false_positive_rate']*100:.2f}%")

    # PyTorch Datasets
    device = torch.device("cpu")
    t_X_tr = torch.tensor(X_tr_sc, dtype=torch.float32)
    t_y_prob_tr = torch.tensor(y_prob_tr, dtype=torch.float32)
    t_y_stage_tr = torch.tensor(y_stage_tr, dtype=torch.long)

    t_X_val = torch.tensor(X_val_sc, dtype=torch.float32)
    t_y_prob_val = torch.tensor(y_prob_val, dtype=torch.float32)
    t_y_stage_val = torch.tensor(y_stage_val, dtype=torch.long)

    t_X_te = torch.tensor(X_te_sc, dtype=torch.float32)
    t_y_prob_te = torch.tensor(y_prob_te, dtype=torch.float32)
    t_y_stage_te = torch.tensor(y_stage_te, dtype=torch.long)

    tr_loader = DataLoader(TensorDataset(t_X_tr, t_y_prob_tr, t_y_stage_tr), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(t_X_val, t_y_prob_val, t_y_stage_val), batch_size=64, shuffle=False)
    te_loader = DataLoader(TensorDataset(t_X_te, t_y_prob_te, t_y_stage_te), batch_size=64, shuffle=False)

    counts = np.bincount(y_stage_tr, minlength=len(STAGE_NAMES))
    weights = [len(y_stage_tr) / (len(STAGE_NAMES) * max(c, 1)) for c in counts]
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    # 2. Standard Attention LSTM Sweep (Tuning stopping epoch STRICTLY on Val split)
    print("\n--- 2. Training Standard Attention LSTM (25 Epochs, Tuned on Validation Split) ---")
    set_seed(42)
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

    epoch_logs = []
    best_val_macro_f1 = -1.0
    best_val_epoch = 1
    best_weights = None

    for ep in range(1, 26):
        model.train()
        running_tr_loss = 0.0
        for bx, bprob, bstage in tr_loader:
            optimizer.zero_grad()
            pred_prob, pred_stage, _ = model(bx)
            loss = bce_loss(pred_prob, bprob) + ce_loss(pred_stage, bstage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_tr_loss += loss.item() * len(bx)

        tr_loss_ep = running_tr_loss / len(t_X_tr)

        # EVALUATE ON VALIDATION SLICE ONLY (NOT TEST)
        val_m = evaluate_dataset(model, val_loader, device, y_stage_val, y_prob_val, model_name="Val Evaluation")
        val_f1 = val_m["f1_score"]
        val_acc = val_m["accuracy"]
        val_t1 = val_m["k_step_forecast"]["horizon_t+1"]["f1_score"]
        val_recon = val_m["per_stage"]["Reconnaissance"]["f1_score"]
        val_cred = val_m["per_stage"]["Credential Access"]["f1_score"]
        val_lat = val_m["per_stage"]["Lateral Movement"]["f1_score"]
        val_fpr = val_m["false_positive_rate"]

        epoch_logs.append({
            "epoch": ep,
            "tr_loss": round(tr_loss_ep, 4),
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_t1": val_t1,
            "val_recon": val_recon,
            "val_cred": val_cred,
            "val_lat": val_lat,
            "val_fpr": val_fpr,
        })

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_val_epoch = ep
            best_weights = copy.deepcopy(model.state_dict())

    # Print Validation Dynamics Table
    print("\n" + "=" * 90)
    print("STANDARD LSTM: VALIDATION SLICE DYNAMICS (EXCLUSIVELY FOR MODEL SELECTION)")
    print("=" * 90)
    print(f"{'Epoch':<6} | {'TrLoss':<7} | {'ValAcc':<7} | {'ValMacroF1':<11} | {'Val-t+1':<8} | {'ValRecon':<9} | {'ValCred':<8} | {'ValLat':<7} | {'ValFPR':<6}")
    print("-" * 90)
    for row in epoch_logs:
        star = " <-- PEAK VAL EPOCH" if row["epoch"] == best_val_epoch else ""
        print(f"{row['epoch']:<6} | {row['tr_loss']:<7.4f} | {row['val_acc']*100:<6.2f}% | {row['val_f1']*100:<10.2f}% | {row['val_t1']*100:<7.2f}% | {row['val_recon']*100:<8.2f}% | {row['val_cred']*100:<7.2f}% | {row['val_lat']*100:<6.2f}% | {row['val_fpr']*100:<5.2f}%{star}")

    print("=" * 90)
    print(f"Optimal Stopping Epoch selected by Validation slice: Epoch {best_val_epoch} (Val Macro F1 = {best_val_macro_f1*100:.2f}%)")

    # 3. Load peak validation checkpoint and evaluate ONCE on untouched Test Slice
    print("\n--- 3. Evaluating Selected Epoch Checkpoint ONCE on Untouched Test Split ---")
    model.load_state_dict(best_weights)
    test_metrics = evaluate_dataset(
        model, te_loader, device, y_stage_te, y_prob_te,
        model_name=f"ChronoGuard LSTM (Selected @ Val Epoch {best_val_epoch})"
    )
    test_metrics["selected_epoch"] = best_val_epoch

    with open("results/lstm_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    torch.save({
        "model_state_dict": best_weights,
        "input_dim": F_dim,
        "hidden_dim": 64,
        "num_layers": 2,
        "forecast_horizon_k": 3,
        "num_stages": 6,
        "best_epoch": best_val_epoch,
    }, "models/chronoguard_lstm.pt")

    # 4. LSTM + SMOTE variant (trained on Train+SMOTE, tuned on Val, tested ONCE on Test)
    print("\n--- 4. Training LSTM + SMOTE (25 Epochs, Tuned on Validation Split) ---")
    set_seed(42)
    X_tr_smote, y_prob_tr_smote, y_stage_tr_smote = sequence_smote(
        X_tr_sc, y_prob_tr, y_stage_tr,
        target_classes={4: 300, 3: 200, 2: 400},
        k_neighbors=5,
    )
    t_X_tr_sm = torch.tensor(X_tr_smote, dtype=torch.float32)
    t_y_prob_tr_sm = torch.tensor(y_prob_tr_smote, dtype=torch.float32)
    t_y_stage_tr_sm = torch.tensor(y_stage_tr_smote, dtype=torch.long)
    tr_sm_loader = DataLoader(TensorDataset(t_X_tr_sm, t_y_prob_tr_sm, t_y_stage_tr_sm), batch_size=64, shuffle=True)

    model_sm = ChronoGuardLSTM(
        input_dim=F_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        forecast_horizon_k=3,
        num_stages=6,
        use_max_pool=False,
    ).to(device)

    opt_sm = optim.Adam(model_sm.parameters(), lr=0.001)
    best_sm_val_f1 = -1.0
    best_sm_val_epoch = 1
    best_sm_weights = None
    sm_epoch_logs = []

    for ep in range(1, 26):
        model_sm.train()
        for bx, bprob, bstage in tr_sm_loader:
            opt_sm.zero_grad()
            pp, ps, _ = model_sm(bx)
            l = bce_loss(pp, bprob) + ce_loss(ps, bstage)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model_sm.parameters(), 1.0)
            opt_sm.step()

        val_m_sm = evaluate_dataset(model_sm, val_loader, device, y_stage_val, y_prob_val, model_name="Val SMOTE")
        sm_epoch_logs.append({
            "epoch": ep,
            "val_acc": val_m_sm["accuracy"],
            "val_f1": val_m_sm["f1_score"],
            "val_t1": val_m_sm["k_step_forecast"]["horizon_t+1"]["f1_score"],
            "val_recon": val_m_sm["per_stage"]["Reconnaissance"]["f1_score"],
            "val_cred": val_m_sm["per_stage"]["Credential Access"]["f1_score"],
            "val_lat": val_m_sm["per_stage"]["Lateral Movement"]["f1_score"],
            "val_fpr": val_m_sm["false_positive_rate"],
        })

        if val_m_sm["f1_score"] > best_sm_val_f1:
            best_sm_val_f1 = val_m_sm["f1_score"]
            best_sm_val_epoch = ep
            best_sm_weights = copy.deepcopy(model_sm.state_dict())

    print(f"Optimal SMOTE Stopping Epoch selected by Validation slice: Epoch {best_sm_val_epoch} (Val Macro F1 = {best_sm_val_f1*100:.2f}%)")

    # Evaluate SMOTE ONCE on Test Split
    model_sm.load_state_dict(best_sm_weights)
    smote_test_metrics = evaluate_dataset(
        model_sm, te_loader, device, y_stage_te, y_prob_te,
        model_name=f"ChronoGuard LSTM + SMOTE (Selected @ Val Epoch {best_sm_val_epoch})"
    )
    smote_test_metrics["selected_epoch"] = best_sm_val_epoch

    with open("results/lstm_smote_metrics.json", "w") as f:
        json.dump(smote_test_metrics, f, indent=2)

    # Consolidated Final Comparison Table on UNTOUCHED Test Split
    print("\n\n" + "=" * 100)
    print("      FINAL UNTOUCHED TEST SET BENCHMARK METRICS (STRICT 3-WAY SPLIT)      ")
    print("=" * 100)
    fmt = "{:<32} | {:>6} | {:>8} | {:>6} | {:>7} | {:>8} | {:>7} | {:>7} | {:>7} | {:>7}"
    print(fmt.format("Model", "Acc", "Macro F1", "FPR", "t+1 F1", "Recon", "CredAcc", "InitAcc", "LatMove", "Impact"))
    print("-" * 100)

    for m in [base_metrics, test_metrics, smote_test_metrics]:
        p = m["per_stage"]
        t1 = f"{m['k_step_forecast']['horizon_t+1']['f1_score']*100:.1f}%" if "k_step_forecast" in m else "N/A"
        print(fmt.format(
            m["model_name"][:32],
            f"{m['accuracy']*100:.1f}%",
            f"{m['f1_score']*100:.1f}%",
            f"{m['false_positive_rate']*100:.1f}%",
            t1,
            f"{p.get('Reconnaissance',{}).get('f1_score',0)*100:.1f}%",
            f"{p.get('Credential Access',{}).get('f1_score',0)*100:.1f}%",
            f"{p.get('Initial Access',{}).get('f1_score',0)*100:.1f}%",
            f"{p.get('Lateral Movement',{}).get('f1_score',0)*100:.1f}%",
            f"{p.get('Impact',{}).get('f1_score',0)*100:.1f}%",
        ))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
