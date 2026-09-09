"""Diagnostic script to inspect Credential Access predictions across 5 seeds with RobustScaler.

Analyzes:
1. Stopping epochs for all 5 seeds (42, 123, 456, 789, 2024).
2. Exact predictions for the 112 Credential Access test sequences for each seed.
3. In failed seeds, what are Credential Access sequences being classified as (Benign, Impact, Recon, etc.)?
4. Statistical aggregation and saving to results/stability_benchmark.json.
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
    all_stage_logits = []

    with torch.no_grad():
        for bx, _, _ in loader:
            bx = bx.to(device)
            pp, ps, _ = model(bx)
            all_prob.append(pp.cpu().numpy())
            all_stage_logits.append(ps.cpu().numpy())

    preds_prob = np.concatenate(all_prob, axis=0)
    preds_logits = np.concatenate(all_stage_logits, axis=0)
    preds_stage = np.argmax(preds_logits, axis=-1)

    acc = float(accuracy_score(y_true_stage, preds_stage))
    macro_f1 = float(f1_score(y_true_stage, preds_stage, average="macro", zero_division=0))
    prec = float(precision_score(y_true_stage, preds_stage, average="macro", zero_division=0))
    rec = float(recall_score(y_true_stage, preds_stage, average="macro", zero_division=0))

    bin_true = (y_true_stage > 0).astype(int)
    bin_pred = (preds_stage > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

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
    t1_acc = float(accuracy_score(t1_true, t1_pred))

    # Multi-horizon metrics
    k_step = {}
    for h in range(forecast_k):
        ht = (y_true_prob[:, h] >= 0.5).astype(int)
        hp = (preds_prob[:, h] >= 0.5).astype(int)
        k_step[f"horizon_t+{h+1}"] = {
            "accuracy": float(accuracy_score(ht, hp)),
            "f1_score": float(f1_score(ht, hp, average="binary", zero_division=0)),
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "fnr": fnr,
        "t1_f1": t1_f1,
        "t1_acc": t1_acc,
        "k_step_forecast": k_step,
        "per_stage": {
            s: {
                "precision": report[s]["precision"],
                "recall": report[s]["recall"],
                "f1_score": report[s]["f1-score"],
                "support": int(report[s]["support"]),
            }
            for s in STAGE_NAMES
            if s in report
        },
        "preds_stage": preds_stage,
        "preds_logits": preds_logits,
    }


def evaluate_baseline(clf, X_scaled, y_true_stage):
    N, W, F = X_scaled.shape
    preds = clf.predict(X_scaled.reshape(N, W * F))
    acc = float(accuracy_score(y_true_stage, preds))
    macro_f1 = float(f1_score(y_true_stage, preds, average="macro", zero_division=0))
    prec = float(precision_score(y_true_stage, preds, average="macro", zero_division=0))
    rec = float(recall_score(y_true_stage, preds, average="macro", zero_division=0))

    bin_true = (y_true_stage > 0).astype(int)
    bin_pred = (preds > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

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
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "fnr": fnr,
        "per_stage": {
            s: {
                "precision": report[s]["precision"],
                "recall": report[s]["recall"],
                "f1_score": report[s]["f1-score"],
                "support": int(report[s]["support"]),
            }
            for s in STAGE_NAMES
            if s in report
        },
        "preds_stage": preds,
    }


def main():
    print("=" * 80)
    print("ChronoGuard Credential Access Diagnostic across 5 Seeds (RobustScaler)")
    print("=" * 80)

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
    N_te = len(y_stage_te)

    # RobustScaler on FULL train
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

    # Credential Access is index 2
    cred_idx = 2
    cred_mask = (y_stage_te == cred_idx)
    num_cred = int(np.sum(cred_mask))
    print(f"Total Test Sequences: {N_te}, of which Credential Access: {num_cred}")

    # Baseline evaluation across seeds
    for seed in seeds:
        set_seed(seed)
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        clf.fit(X_tr_sc.reshape(N_tr, W * F_dim), y_stage_tr)
        bl_eval = evaluate_baseline(clf, X_te_sc, y_stage_te)
        bl_eval["seed"] = seed
        baseline_results.append(bl_eval)

    # Check baseline predictions for Credential Access
    bl_cred_preds = baseline_results[0]["preds_stage"][cred_mask]
    bl_cred_dist = {STAGE_NAMES[i]: int(np.sum(bl_cred_preds == i)) for i in range(len(STAGE_NAMES))}
    print(f"\nBaseline Credential Access Predictions Distribution (N={num_cred}):")
    for stg, count in bl_cred_dist.items():
        print(f"  {stg:<20}: {count:>3} ({count/num_cred*100:.1f}%)")

    # Detailed LSTM runs
    detailed_seed_diagnostics = []

    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n--- Training LSTM Seed {seed} ({run_idx}/5) ---")
        set_seed(seed)

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
        epoch_history = []

        for ep in range(1, 16):
            model.train()
            total_train_loss = 0.0
            for bx, bprob, bstage in tr_loader:
                optimizer.zero_grad()
                pp, ps, _ = model(bx)
                l = bce_loss(pp, bprob) + ce_loss(ps, bstage)
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train_loss += l.item()

            val_res = evaluate_loader(model, val_loader, device, y_stage_val, y_prob_val)
            val_f1 = val_res["macro_f1"]
            val_cred_f1 = val_res["per_stage"]["Credential Access"]["f1_score"]

            epoch_history.append({
                "epoch": ep,
                "train_loss": total_train_loss / len(tr_loader),
                "val_macro_f1": val_f1,
                "val_cred_f1": val_cred_f1,
            })

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
        test_eval["epoch_history"] = epoch_history
        lstm_results.append(test_eval)

        # Inspect the 112 Credential Access predictions specifically
        cred_preds = test_eval["preds_stage"][cred_mask]
        cred_breakdown = {STAGE_NAMES[i]: int(np.sum(cred_preds == i)) for i in range(len(STAGE_NAMES))}
        
        diag = {
            "seed": seed,
            "selected_epoch": best_val_ep,
            "val_macro_f1": best_val_f1,
            "test_macro_f1": test_eval["macro_f1"],
            "test_cred_f1": test_eval["per_stage"]["Credential Access"]["f1_score"],
            "test_recon_f1": test_eval["per_stage"]["Reconnaissance"]["f1_score"],
            "test_t1_f1": test_eval["t1_f1"],
            "cred_breakdown": cred_breakdown,
        }
        detailed_seed_diagnostics.append(diag)

        print(f"  Selected Epoch: {best_val_ep} (Val F1: {best_val_f1*100:.2f}%)")
        print(f"  Test Macro F1: {test_eval['macro_f1']*100:.2f}%, Cred F1: {diag['test_cred_f1']*100:.2f}%")
        print(f"  Credential Access (N={num_cred}) Predicted As:")
        for stg, cnt in cred_breakdown.items():
            if cnt > 0:
                print(f"    - {stg}: {cnt} ({cnt/num_cred*100:.1f}%)")

    # Output detailed comparative diagnosis
    print("\n" + "=" * 90)
    print("DETAILED CREDENTIAL ACCESS (N=112) PREDICTIONS BY SEED")
    print("=" * 90)
    header = f"{'Seed':<6} | {'StopEp':<6} | {'ValMacroF1':<10} | {'TestMacroF1':<11} | {'Cred F1':<8} | {'Pred: Cred':<10} | {'Pred: Benign':<12} | {'Pred: Impact':<12} | {'Pred: Recon/Init/Lat'}"
    print(header)
    print("-" * 90)
    for d in detailed_seed_diagnostics:
        cb = d["cred_breakdown"]
        other = cb["Reconnaissance"] + cb["Initial Access"] + cb["Lateral Movement"]
        row = f"{d['seed']:<6} | Ep {d['selected_epoch']:<3} | {d['val_macro_f1']*100:>8.2f}% | {d['test_macro_f1']*100:>9.2f}% | {d['test_cred_f1']*100:>6.2f}% | {cb['Credential Access']:>8} ({cb['Credential Access']/num_cred*100:4.1f}%) | {cb['Benign']:>10} ({cb['Benign']/num_cred*100:4.1f}%) | {cb['Impact']:>10} ({cb['Impact']/num_cred*100:4.1f}%) | {other:>4}"
        print(row)
    print("=" * 90)

    # Save comprehensive stability_benchmark.json
    bl_macro_f1s = [r["macro_f1"] * 100 for r in baseline_results]
    bl_accuracies = [r["accuracy"] * 100 for r in baseline_results]
    bl_fprs = [r["fpr"] * 100 for r in baseline_results]
    bl_fnrs = [r["fnr"] * 100 for r in baseline_results]

    lstm_macro_f1s = [r["macro_f1"] * 100 for r in lstm_results]
    lstm_accuracies = [r["accuracy"] * 100 for r in lstm_results]
    lstm_t1_f1s = [r["t1_f1"] * 100 for r in lstm_results]
    lstm_t1_accs = [r["t1_acc"] * 100 for r in lstm_results]
    lstm_fprs = [r["fpr"] * 100 for r in lstm_results]
    lstm_fnrs = [r["fnr"] * 100 for r in lstm_results]

    # Per-stage aggregations
    stages_summary = {}
    for stg in STAGE_NAMES:
        bl_stg_f1 = [r["per_stage"][stg]["f1_score"] * 100 for r in baseline_results]
        lstm_stg_f1 = [r["per_stage"][stg]["f1_score"] * 100 for r in lstm_results]
        stages_summary[stg] = {
            "baseline": {
                "mean_f1": round(float(np.mean(bl_stg_f1)), 2),
                "std_f1": round(float(np.std(bl_stg_f1)), 2),
            },
            "lstm": {
                "mean_f1": round(float(np.mean(lstm_stg_f1)), 2),
                "std_f1": round(float(np.std(lstm_stg_f1)), 2),
                "min_f1": round(float(np.min(lstm_stg_f1)), 2),
                "max_f1": round(float(np.max(lstm_stg_f1)), 2),
            },
        }

    # Forecast horizons aggregation
    horizons = ["horizon_t+1", "horizon_t+2", "horizon_t+3"]
    horizon_summary = {}
    for h in horizons:
        h_accs = [r["k_step_forecast"][h]["accuracy"] * 100 for r in lstm_results]
        h_f1s = [r["k_step_forecast"][h]["f1_score"] * 100 for r in lstm_results]
        horizon_summary[h] = {
            "mean_accuracy": round(float(np.mean(h_accs)), 2),
            "std_accuracy": round(float(np.std(h_accs)), 2),
            "mean_f1": round(float(np.mean(h_f1s)), 2),
            "std_f1": round(float(np.std(h_f1s)), 2),
        }

    benchmark_data = {
        "scaler": "RobustScaler (Full Train + 5-sigma clip)",
        "split": "3-way chronological (65% Train, 10% Val, 25% Test)",
        "total_test_samples": N_te,
        "seeds_evaluated": seeds,
        "baseline": {
            "mean_macro_f1": round(float(np.mean(bl_macro_f1s)), 2),
            "std_macro_f1": round(float(np.std(bl_macro_f1s)), 2),
            "mean_accuracy": round(float(np.mean(bl_accuracies)), 2),
            "std_accuracy": round(float(np.std(bl_accuracies)), 2),
            "mean_fpr": round(float(np.mean(bl_fprs)), 2),
            "std_fpr": round(float(np.std(bl_fprs)), 2),
            "mean_fnr": round(float(np.mean(bl_fnrs)), 2),
            "std_fnr": round(float(np.std(bl_fnrs)), 2),
            "stages": {s: stages_summary[s]["baseline"] for s in STAGE_NAMES},
        },
        "lstm": {
            "mean_macro_f1": round(float(np.mean(lstm_macro_f1s)), 2),
            "std_macro_f1": round(float(np.std(lstm_macro_f1s)), 2),
            "range_macro_f1": [round(float(np.min(lstm_macro_f1s)), 2), round(float(np.max(lstm_macro_f1s)), 2)],
            "mean_accuracy": round(float(np.mean(lstm_accuracies)), 2),
            "std_accuracy": round(float(np.std(lstm_accuracies)), 2),
            "mean_fpr": round(float(np.mean(lstm_fprs)), 2),
            "std_fpr": round(float(np.std(lstm_fprs)), 2),
            "mean_fnr": round(float(np.mean(lstm_fnrs)), 2),
            "std_fnr": round(float(np.std(lstm_fnrs)), 2),
            "selected_epochs": [r["selected_epoch"] for r in lstm_results],
            "horizons": horizon_summary,
            "stages": {s: stages_summary[s]["lstm"] for s in STAGE_NAMES},
            "runs": [
                {
                    "seed": r["seed"],
                    "selected_epoch": r["selected_epoch"],
                    "val_macro_f1": round(r["val_macro_f1"] * 100, 2),
                    "test_macro_f1": round(r["macro_f1"] * 100, 2),
                    "test_accuracy": round(r["accuracy"] * 100, 2),
                    "t1_f1": round(r["t1_f1"] * 100, 2),
                    "recon_f1": round(r["per_stage"]["Reconnaissance"]["f1_score"] * 100, 2),
                    "cred_f1": round(r["per_stage"]["Credential Access"]["f1_score"] * 100, 2),
                    "init_f1": round(r["per_stage"]["Initial Access"]["f1_score"] * 100, 2),
                    "lat_f1": round(r["per_stage"]["Lateral Movement"]["f1_score"] * 100, 2),
                    "impact_f1": round(r["per_stage"]["Impact"]["f1_score"] * 100, 2),
                }
                for r in lstm_results
            ],
            "credential_access_diagnostics": detailed_seed_diagnostics,
        },
    }

    out_path = "results/stability_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nSaved updated benchmark to {out_path}")


if __name__ == "__main__":
    main()
