"""Diagnostic script for Initial Access breakdown and fresh CPU latency / footprint benchmarking."""

import os
import sys
import time
import copy
import json
import yaml
import random
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score

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
        "t1_f1": t1_f1,
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
    }


def main():
    print("=" * 80)
    print("1. INITIAL ACCESS DIAGNOSTIC & BREAKDOWN ACROSS 5 SEEDS")
    print("=" * 80)

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sel = cfg["dataset"]["selected_features"]
    sample_per_file = cfg["dataset"]["sample_per_file"]

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

    # Initial Access is index 3
    init_idx = 3
    init_mask = (y_stage_te == init_idx)
    num_init = int(np.sum(init_mask))
    print(f"Total Test Sequences: {N_te}, of which Initial Access: {num_init}")

    # Baseline evaluation
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(X_tr_sc.reshape(N_tr, W * F_dim), y_stage_tr)
    bl_preds = clf.predict(X_te_sc.reshape(N_te, W * F_dim))
    bl_init_preds = bl_preds[init_mask]
    bl_init_dist = {STAGE_NAMES[i]: int(np.sum(bl_init_preds == i)) for i in range(len(STAGE_NAMES))}
    bl_init_f1 = f1_score(y_stage_te == init_idx, bl_preds == init_idx, zero_division=0)

    print(f"\nBaseline Initial Access F1: {bl_init_f1*100:.2f}%")
    print("Baseline Predictions for Initial Access (N=27):")
    for stg, cnt in bl_init_dist.items():
        if cnt > 0:
            print(f"  - {stg}: {cnt} ({cnt/num_init*100:.1f}%)")

    # Evaluate 5 seeds
    seed_init_diagnostics = []
    trained_models = {}

    for seed in seeds:
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
            if val_res["macro_f1"] > best_val_f1:
                best_val_f1 = val_res["macro_f1"]
                best_val_ep = ep
                best_weights = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_weights)
        test_eval = evaluate_loader(model, te_loader, device, y_stage_te, y_prob_te)
        trained_models[seed] = (model, best_weights)

        # Inspect the 27 Initial Access predictions
        init_preds = test_eval["preds_stage"][init_mask]
        init_breakdown = {STAGE_NAMES[i]: int(np.sum(init_preds == i)) for i in range(len(STAGE_NAMES))}
        init_f1 = test_eval["per_stage"]["Initial Access"]["f1_score"]

        seed_init_diagnostics.append({
            "seed": seed,
            "selected_epoch": best_val_ep,
            "val_macro_f1": best_val_f1,
            "test_macro_f1": test_eval["macro_f1"],
            "test_init_f1": init_f1,
            "test_cred_f1": test_eval["per_stage"]["Credential Access"]["f1_score"],
            "init_breakdown": init_breakdown,
        })

    print("\n" + "=" * 105)
    print("DETAILED INITIAL ACCESS (N=27) PREDICTIONS BY SEED")
    print("=" * 105)
    fmt = "{:<6} | {:<6} | {:>10} | {:>11} | {:>9} | {:>12} | {:>12} | {:>12} | {:>14}"
    print(fmt.format("Seed", "StopEp", "ValMacroF1", "TestMacroF1", "Init F1", "Pred: InitAcc", "Pred: Benign", "Pred: Impact", "Pred: LatMove/Other"))
    print("-" * 105)
    for d in seed_init_diagnostics:
        ib = d["init_breakdown"]
        other = ib["Reconnaissance"] + ib["Credential Access"] + ib["Lateral Movement"]
        print(fmt.format(
            d["seed"],
            f"Ep {d['selected_epoch']}",
            f"{d['val_macro_f1']*100:.2f}%",
            f"{d['test_macro_f1']*100:.2f}%",
            f"{d['test_init_f1']*100:.2f}%",
            f"{ib['Initial Access']} ({ib['Initial Access']/num_init*100:.1f}%)",
            f"{ib['Benign']} ({ib['Benign']/num_init*100:.1f}%)",
            f"{ib['Impact']} ({ib['Impact']/num_init*100:.1f}%)",
            f"{other}",
        ))
    print("=" * 105)

    # 2. FRESH CPU INFERENCE LATENCY & MEMORY BENCHMARK
    print("\n" + "=" * 80)
    print("2. FRESH CPU INFERENCE LATENCY & FOOTPRINT BENCHMARK")
    print("=" * 80)

    # Save the Seed 42 model and scaler as canonical
    best_model, best_state = trained_models[42]
    torch.save(best_state, "models/chronoguard_lstm.pt")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(clf, "models/baseline_lr.pkl")

    # File sizes
    lstm_size = os.path.getsize("models/chronoguard_lstm.pt")
    bl_size = os.path.getsize("models/baseline_lr.pkl")
    scaler_size = os.path.getsize("models/scaler.pkl")

    print(f"File Size on Disk:")
    print(f"  - models/chronoguard_lstm.pt: {lstm_size:,} bytes ({lstm_size/1024:.1f} KB)")
    print(f"  - models/baseline_lr.pkl:     {bl_size:,} bytes ({bl_size/1024:.1f} KB)")
    print(f"  - models/scaler.pkl:          {scaler_size:,} bytes ({scaler_size/1024:.1f} KB)")

    # Latency benchmark
    # Single sample: (1, 10, 16)
    dummy_sample = torch.randn(1, 10, 16, dtype=torch.float32)
    dummy_baseline_sample = np.random.randn(1, 10 * 16)

    # Warmup
    best_model.eval()
    with torch.no_grad():
        for _ in range(50):
            _ = best_model(dummy_sample)
            _ = clf.predict(dummy_baseline_sample)

    # Measure LSTM single-window latency over 1,000 iterations
    N_RUNS = 1000
    lstm_latencies = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = best_model(dummy_sample)
            t1 = time.perf_counter()
            lstm_latencies.append((t1 - t0) * 1000.0)  # ms

    # Measure Baseline single-window latency over 1,000 iterations
    bl_latencies = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = clf.predict(dummy_baseline_sample)
        t1 = time.perf_counter()
        bl_latencies.append((t1 - t0) * 1000.0)  # ms

    lstm_mean_lat = np.mean(lstm_latencies)
    lstm_std_lat = np.std(lstm_latencies)
    lstm_p95_lat = np.percentile(lstm_latencies, 95)
    lstm_p99_lat = np.percentile(lstm_latencies, 99)

    bl_mean_lat = np.mean(bl_latencies)
    bl_std_lat = np.std(bl_latencies)
    bl_p95_lat = np.percentile(bl_latencies, 95)
    bl_p99_lat = np.percentile(bl_latencies, 99)

    print(f"\nSingle-Window CPU Inference Latency (N={N_RUNS} runs):")
    print(f"  Baseline (Logistic Regression):")
    print(f"    - Mean: {bl_mean_lat:.3f} ms ± {bl_std_lat:.3f} ms")
    print(f"    - P95:  {bl_p95_lat:.3f} ms")
    print(f"    - P99:  {bl_p99_lat:.3f} ms")
    print(f"  ChronoGuard (LSTM + Attention):")
    print(f"    - Mean: {lstm_mean_lat:.3f} ms ± {lstm_std_lat:.3f} ms")
    print(f"    - P95:  {lstm_p95_lat:.3f} ms")
    print(f"    - P99:  {lstm_p99_lat:.3f} ms")

    # Save benchmark update dictionary
    fresh_benchmarks = {
        "footprint": {
            "baseline_bytes": bl_size,
            "baseline_kb": round(bl_size / 1024, 1),
            "lstm_bytes": lstm_size,
            "lstm_kb": round(lstm_size / 1024, 1),
            "scaler_bytes": scaler_size,
            "scaler_kb": round(scaler_size / 1024, 1),
        },
        "latency_ms": {
            "baseline_mean": round(bl_mean_lat, 2),
            "baseline_std": round(bl_std_lat, 2),
            "baseline_p99": round(bl_p99_lat, 2),
            "lstm_mean": round(lstm_mean_lat, 2),
            "lstm_std": round(lstm_std_lat, 2),
            "lstm_p99": round(lstm_p99_lat, 2),
        },
        "initial_access_diagnostics": seed_init_diagnostics,
    }

    with open("results/fresh_benchmarks.json", "w") as f:
        json.dump(fresh_benchmarks, f, indent=2)
    print("\nSaved fresh benchmarks to results/fresh_benchmarks.json")


if __name__ == "__main__":
    main()
