"""Accurate CPU inference latency and footprint measurement on current saved models."""

import os
import sys
import time
import json
import joblib
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.lstm_model import ChronoGuardLSTM


def main():
    print("=" * 80)
    print("BENCHMARKING CPU INFERENCE LATENCY AND MODEL FOOTPRINT")
    print("=" * 80)

    lstm_path = "models/chronoguard_lstm.pt"
    bl_path = "models/baseline_lr.pkl"
    scaler_path = "models/scaler.pkl"

    # 1. Exact File Sizes
    lstm_bytes = os.path.getsize(lstm_path)
    bl_bytes = os.path.getsize(bl_path)
    scaler_bytes = os.path.getsize(scaler_path)

    print("\nModel Footprint on Disk:")
    print(f"  - ChronoGuard LSTM ({lstm_path}): {lstm_bytes:,} bytes ({lstm_bytes / 1024:.1f} KB)")
    print(f"  - Baseline LR ({bl_path}):          {bl_bytes:,} bytes ({bl_bytes / 1024:.1f} KB)")
    print(f"  - Scaler ({scaler_path}):                {scaler_bytes:,} bytes ({scaler_bytes / 1024:.1f} KB)")

    # 2. Load Models
    clf = joblib.load(bl_path)
    scaler = joblib.load(scaler_path)

    # Determine input dim
    # Logistic regression n_features_in_
    bl_features = clf.n_features_in_  # W * F_dim
    W = 10
    F_dim = bl_features // W
    print(f"\nModel Dimensions: W={W}, Features per window={F_dim}, Flattened baseline={bl_features}")

    model = ChronoGuardLSTM(
        input_dim=F_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        forecast_horizon_k=3,
        num_stages=6,
        use_max_pool=False,
    )
    state_dict = torch.load(lstm_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ChronoGuard LSTM Total Parameters: {total_params:,} ({trainable_params:,} trainable)")

    # 3. CPU Latency Benchmark (Single Window: Batch Size = 1)
    x_torch = torch.randn(1, W, F_dim, dtype=torch.float32)
    x_numpy = np.random.randn(1, W * F_dim)

    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = model(x_torch)
            _ = clf.predict(x_numpy)

    N_RUNS = 1000

    # Benchmark LSTM (single window)
    lstm_times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = model(x_torch)
            t1 = time.perf_counter()
            lstm_times.append((t1 - t0) * 1000.0)  # ms

    # Benchmark Baseline (single window)
    bl_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = clf.predict(x_numpy)
        t1 = time.perf_counter()
        bl_times.append((t1 - t0) * 1000.0)  # ms

    # 4. Batch Inference (e.g., Batch Size = 64) per-sample latency
    BATCH_SIZE = 64
    x_batch_torch = torch.randn(BATCH_SIZE, W, F_dim, dtype=torch.float32)
    x_batch_numpy = np.random.randn(BATCH_SIZE, W * F_dim)

    lstm_batch_times = []
    with torch.no_grad():
        for _ in range(200):
            t0 = time.perf_counter()
            _ = model(x_batch_torch)
            t1 = time.perf_counter()
            lstm_batch_times.append(((t1 - t0) * 1000.0) / BATCH_SIZE)  # ms per sample

    bl_batch_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = clf.predict(x_batch_numpy)
        t1 = time.perf_counter()
        bl_batch_times.append(((t1 - t0) * 1000.0) / BATCH_SIZE)  # ms per sample

    print(f"\n--- Fresh Single-Window CPU Inference Latency (N={N_RUNS} runs, Batch Size = 1) ---")
    print(f"  Baseline (Logistic Regression):")
    print(f"    - Mean: {np.mean(bl_times):.3f} ms ± {np.std(bl_times):.3f} ms")
    print(f"    - 50th percentile (Median): {np.percentile(bl_times, 50):.3f} ms")
    print(f"    - 95th percentile:          {np.percentile(bl_times, 95):.3f} ms")
    print(f"    - 99th percentile:          {np.percentile(bl_times, 99):.3f} ms")
    print(f"  ChronoGuard (LSTM + Attention):")
    print(f"    - Mean: {np.mean(lstm_times):.3f} ms ± {np.std(lstm_times):.3f} ms")
    print(f"    - 50th percentile (Median): {np.percentile(lstm_times, 50):.3f} ms")
    print(f"    - 95th percentile:          {np.percentile(lstm_times, 95):.3f} ms")
    print(f"    - 99th percentile:          {np.percentile(lstm_times, 99):.3f} ms")

    print(f"\n--- Fresh Batch CPU Inference Latency (Batch Size = {BATCH_SIZE}, per-sample time) ---")
    print(f"  Baseline per window:   {np.mean(bl_batch_times):.3f} ms / window")
    print(f"  ChronoGuard per window: {np.mean(lstm_batch_times):.3f} ms / window")

    # Save to json
    results = {
        "model_footprint": {
            "baseline_lr_pkl_bytes": bl_bytes,
            "baseline_lr_pkl_kb": round(bl_bytes / 1024, 1),
            "chronoguard_lstm_pt_bytes": lstm_bytes,
            "chronoguard_lstm_pt_kb": round(lstm_bytes / 1024, 1),
            "scaler_pkl_bytes": scaler_bytes,
            "scaler_pkl_kb": round(scaler_bytes / 1024, 1),
            "lstm_parameters": total_params,
        },
        "single_window_latency_ms": {
            "baseline": {
                "mean": round(float(np.mean(bl_times)), 3),
                "std": round(float(np.std(bl_times)), 3),
                "median": round(float(np.percentile(bl_times, 50)), 3),
                "p95": round(float(np.percentile(bl_times, 95)), 3),
                "p99": round(float(np.percentile(bl_times, 99)), 3),
            },
            "lstm": {
                "mean": round(float(np.mean(lstm_times)), 3),
                "std": round(float(np.std(lstm_times)), 3),
                "median": round(float(np.percentile(lstm_times, 50)), 3),
                "p95": round(float(np.percentile(lstm_times, 95)), 3),
                "p99": round(float(np.percentile(lstm_times, 99)), 3),
            },
        },
        "batch_latency_ms_per_window": {
            "baseline": round(float(np.mean(bl_batch_times)), 3),
            "lstm": round(float(np.mean(lstm_batch_times)), 3),
        },
    }

    with open("results/latency_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved fresh latency metrics to results/latency_benchmark.json")


if __name__ == "__main__":
    main()
