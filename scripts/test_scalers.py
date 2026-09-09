"""Diagnostic script to inspect feature scaling and test RobustScaler variants."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from src.data.load_raw import load_raw_dataset
from src.data.windowing import bucket_flows_into_windows, create_sequences, get_feature_names
from src.data.label_mapping import STAGE_NAMES


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    sel = cfg["dataset"]["selected_features"]
    sample_per_file = cfg["dataset"]["sample_per_file"]

    print("Loading 3-way split datasets...")
    tr_df = load_raw_dataset("data/raw", split="train", sample_per_file=sample_per_file, selected_features=sel)
    te_df = load_raw_dataset("data/raw", split="test", sample_per_file=sample_per_file, selected_features=sel)

    w_tr = bucket_flows_into_windows(tr_df, sel, 20)
    w_te = bucket_flows_into_windows(te_df, sel, 20)

    X_tr, y_prob_tr, y_stage_tr, _ = create_sequences(w_tr, 10, 3)
    X_te, y_prob_te, y_stage_te, _ = create_sequences(w_te, 10, 3)

    N_tr, W, F_dim = X_tr.shape
    N_te = X_te.shape[0]

    feature_names = get_feature_names(sel)

    X_tr_flat = X_tr.reshape(-1, F_dim)

    # Scaler 1: Current StandardScaler (all train)
    s1 = StandardScaler().fit(X_tr_flat)

    # Scaler 2: RobustScaler (benign-only train) + 5-sigma clip (as in technical spec)
    benign_idx = np.where(y_stage_tr == 0)[0]
    benign_windows = X_tr[benign_idx].reshape(-1, F_dim)
    s2 = RobustScaler().fit(benign_windows)

    # Scaler 3: RobustScaler (FULL train) + 5-sigma clip
    s3 = RobustScaler().fit(X_tr_flat)

    scalers = {
        "1. StandardScaler (Full Train, Current)": (s1, False),
        "2. RobustScaler (Benign-Only + 5-sigma clip, Spec)": (s2, True),
        "3. RobustScaler (Full Train + 5-sigma clip)": (s3, True),
        "4. RobustScaler (Full Train, No clip)": (s3, False),
    }

    # 1. Feature inspection: examine sample scaled feature vectors for Credential Access vs Recon vs Benign
    print("\n" + "=" * 90)
    print("FEATURE VECTOR INSPECTION: Benign vs Recon vs Credential Access")
    print("=" * 90)

    for name, (scaler, clip) in scalers.items():
        print(f"\n--- {name} ---")
        for stage_id, stage_name in [(0, "Benign"), (1, "Reconnaissance"), (2, "Credential Access")]:
            idx = np.where(y_stage_tr == stage_id)[0]
            if len(idx) > 0:
                sample_feat = scaler.transform(X_tr[idx[:5], -1, :]) # 5 sample windows, last step
                if clip:
                    sample_feat = np.clip(sample_feat, -5.0, 5.0)

                # Check clipping saturation
                sat_pos = np.mean(sample_feat >= 4.99) * 100
                sat_neg = np.mean(sample_feat <= -4.99) * 100
                print(f"  Stage {stage_id} ({stage_name:<18}): Min={np.min(sample_feat):>6.2f}, Max={np.max(sample_feat):>6.2f}, Mean={np.mean(sample_feat):>6.2f} | Saturation [+5s: {sat_pos:>4.1f}%, -5s: {sat_neg:>4.1f}%]")

    # 2. Evaluate Baseline under each Scaler
    print("\n\n" + "=" * 90)
    print("BASELINE PERFORMANCE UNDER DIFFERENT SCALERS (UNTUCHED TEST SET, N=1,805)")
    print("=" * 90)
    fmt = "{:<45} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}"
    print(fmt.format("Scaler Configuration", "Acc", "Macro F1", "Recon F1", "Cred F1", "Init F1", "Impact F1"))
    print("-" * 90)

    for name, (scaler, clip) in scalers.items():
        X_tr_sc = np.zeros_like(X_tr)
        X_te_sc = np.zeros_like(X_te)
        for i in range(W):
            sc_tr = scaler.transform(X_tr[:, i, :])
            sc_te = scaler.transform(X_te[:, i, :])
            if clip:
                sc_tr = np.clip(sc_tr, -5.0, 5.0)
                sc_te = np.clip(sc_te, -5.0, 5.0)
            X_tr_sc[:, i, :] = sc_tr
            X_te_sc[:, i, :] = sc_te

        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        clf.fit(X_tr_sc.reshape(N_tr, W * F_dim), y_stage_tr)
        preds = clf.predict(X_te_sc.reshape(N_te, W * F_dim))

        rep = classification_report(y_stage_te, preds, labels=list(range(6)), target_names=STAGE_NAMES, output_dict=True, zero_division=0)
        macro_f1 = f1_score(y_stage_te, preds, average="macro", zero_division=0)
        acc = np.mean(preds == y_stage_te)

        recon_f1 = rep["Reconnaissance"]["f1-score"] * 100
        cred_f1 = rep["Credential Access"]["f1-score"] * 100
        init_f1 = rep["Initial Access"]["f1-score"] * 100
        imp_f1 = rep["Impact"]["f1-score"] * 100

        print(fmt.format(name[:45], f"{acc*100:.1f}%", f"{macro_f1*100:.2f}%", f"{recon_f1:.1f}%", f"{cred_f1:.1f}%", f"{init_f1:.1f}%", f"{imp_f1:.1f}%"))

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
