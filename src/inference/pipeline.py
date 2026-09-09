"""Unified inference pipeline for ChronoGuard.

Executes the shared preprocessing, scaling, LSTM forecasting, explainability,
and baseline comparison on an uploaded network flow CSV file.
Strictly ensures zero training-serving skew by sharing code with training modules.
"""

import os
import json
import joblib
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.data.clean import clean_dataframe
from src.data.windowing import (
    bucket_flows_into_windows,
    create_sequences,
    get_feature_names,
)
from src.data.label_mapping import ID_TO_STAGE, STAGE_METADATA, is_attack_stage
from src.models.lstm_model import ChronoGuardLSTM
from src.models.explain import explain_prediction_attention


# Cache loaded models in-memory to prevent repeated disk I/O on every request
_MODEL_CACHE: Dict[str, Any] = {}


def load_inference_artifacts(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and cache trained PyTorch model, baseline, scaler, and config."""
    global _MODEL_CACHE
    if "loaded" in _MODEL_CACHE and _MODEL_CACHE["loaded"]:
        return _MODEL_CACHE

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    scaler_path = cfg["inference"]["scaler_path"]
    model_path = cfg["inference"]["model_path"]
    baseline_path = cfg["inference"]["baseline_path"]

    # 1. Scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Fitted scaler not found at {scaler_path}. Run training first.")
    scaler = joblib.load(scaler_path)

    # 2. Baseline Model
    baseline_model = None
    if os.path.exists(baseline_path):
        baseline_model = joblib.load(baseline_path)

    # 3. PyTorch LSTM Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained LSTM weights not found at {model_path}. Run training first.")

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    lstm_model = ChronoGuardLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        forecast_horizon_k=checkpoint["forecast_horizon_k"],
        num_stages=checkpoint["num_stages"],
    )
    lstm_model.load_state_dict(checkpoint["model_state_dict"])
    lstm_model.eval()

    _MODEL_CACHE = {
        "cfg": cfg,
        "scaler": scaler,
        "baseline": baseline_model,
        "lstm": lstm_model,
        "loaded": True,
    }
    return _MODEL_CACHE


def run_pipeline(
    csv_path: str,
    job_id: str = "local_run",
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Execute end-to-end inference on a flow CSV file.
    
    Args:
        csv_path: Path to the uploaded CSV file
        job_id: Unique job ID
        config_path: Path to system config
        
    Returns:
        Structured dictionary conforming to AI-INSTRUCTIONS/04-database-schema.md Section 3.
    """
    artifacts = load_inference_artifacts(config_path)
    cfg = artifacts["cfg"]
    scaler = artifacts["scaler"]
    baseline = artifacts["baseline"]
    lstm_model = artifacts["lstm"]

    selected_features = cfg["dataset"]["selected_features"]
    window_size_flows = cfg["windowing"]["window_size_flows"]
    seq_w = cfg["windowing"]["sequence_length_w"]
    forecast_k = cfg["windowing"]["forecast_horizon_k"]

    # 1. Load and clean uploaded flows
    df_raw = pd.read_csv(csv_path, low_memory=False)
    df_clean = clean_dataframe(df_raw, selected_features=selected_features)

    if len(df_clean) == 0:
        raise ValueError("Uploaded CSV contains no valid flow records after cleaning.")

    # 2. Windowing
    windows = bucket_flows_into_windows(df_clean, selected_features, window_size_flows)
    if not windows:
        raise ValueError("Flow records could not be aggregated into windows.")

    # 3. Sequence creation
    X_seq, _, _, meta = create_sequences(windows, seq_w, forecast_k)
    num_sequences, W, F_dim = X_seq.shape

    # 4. Standard scale features using saved scaler
    X_scaled = np.zeros_like(X_seq)
    for i in range(W):
        X_scaled[:, i, :] = scaler.transform(X_seq[:, i, :])

    feature_names = get_feature_names(selected_features)

    # 5. Baseline Predictions
    baseline_probs = []
    if baseline is not None:
        try:
            X_flat = X_scaled.reshape(num_sequences, W * F_dim)
            if hasattr(baseline, "predict_proba"):
                # Probability of non-benign classes
                b_probs = baseline.predict_proba(X_flat)
                # Sum of non-benign probabilities
                for p_row in b_probs:
                    if len(p_row) > 1:
                        baseline_probs.append(float(np.sum(p_row[1:])))
                    else:
                        baseline_probs.append(0.0)
            else:
                baseline_preds = baseline.predict(X_flat)
                baseline_probs = [1.0 if p > 0 else 0.0 for p in baseline_preds]
        except Exception as e:
            baseline_probs = [0.0] * num_sequences
    else:
        baseline_probs = [0.0] * num_sequences

    # 6. LSTM Forecaster Forward Pass
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        prob_forecast, stage_logits, attention_weights = lstm_model(tensor_x)
        stage_probs = torch.softmax(stage_logits, dim=-1).cpu().numpy()
        pred_stage_ids = torch.argmax(stage_logits, dim=-1).cpu().numpy()
        prob_forecast_np = prob_forecast.cpu().numpy()
        attention_np = attention_weights.cpu().numpy()

    # 7. Format Output Schema conforming to docs/04
    window_predictions = []
    max_prob = 0.0
    stage_counter: Dict[str, int] = {}
    flagged_count = 0
    risk_threshold = cfg["inference"]["risk_thresholds"]["warning"]

    # We produce one output item per sequence/window
    for idx in range(num_sequences):
        inf_prob = float(prob_forecast_np[idx, 0])  # immediate next window probability
        stage_id = int(pred_stage_ids[idx])
        stage_name = ID_TO_STAGE.get(stage_id, "Benign")
        b_prob = float(baseline_probs[idx]) if idx < len(baseline_probs) else 0.0

        if inf_prob > max_prob:
            max_prob = inf_prob
        if inf_prob >= risk_threshold:
            flagged_count += 1

        stage_counter[stage_name] = stage_counter.get(stage_name, 0) + 1

        # Explainability payload
        explain_data = explain_prediction_attention(
            attention_weights=attention_np[idx],
            sequence_features=X_scaled[idx],
            feature_names=feature_names,
            top_k_features=4,
        )

        w_meta = meta[idx] if idx < len(meta) else {}
        w_target_idx = w_meta.get("target_window_index", idx + seq_w)

        window_predictions.append({
            "window_index": idx,
            "target_window_index": w_target_idx,
            "window_start": str(windows[min(idx, len(windows)-1)].get("window_start", f"+{idx*10}s")),
            "window_end": str(windows[min(idx, len(windows)-1)].get("window_end", f"+{(idx+1)*10}s")),
            "infiltration_probability": round(inf_prob, 4),
            "forecast_horizons": [round(float(p), 4) for p in prob_forecast_np[idx]],
            "predicted_stage": stage_name,
            "stage_id": stage_id,
            "stage_metadata": STAGE_METADATA.get(stage_name, STAGE_METADATA["Benign"]),
            "baseline_probability": round(b_prob, 4),
            "top_features": explain_data["top_features"],
            "attention_timeline": explain_data["window_attention"],
            "peak_attention_window": explain_data["peak_window_offset"],
        })

    # Dominant attack stage (or Benign if none flagged)
    attack_stages = {k: v for k, v in stage_counter.items() if is_attack_stage(k)}
    if attack_stages:
        dominant_stage = max(attack_stages, key=attack_stages.get)
    else:
        dominant_stage = "Benign"

    # Overall latest attention weights for the final forecast
    latest_attention = window_predictions[-1]["attention_timeline"] if window_predictions else []
    latest_top_features = window_predictions[-1]["top_features"] if window_predictions else []

    # Model evaluation metrics from disk if available
    comparison_metrics = {}
    if os.path.exists("results/baseline_metrics.json") and os.path.exists("results/lstm_metrics.json"):
        try:
            with open("results/baseline_metrics.json", "r") as fb:
                b_metrics = json.load(fb)
            with open("results/lstm_metrics.json", "r") as fl:
                l_metrics = json.load(fl)
            comparison_metrics = {
                "baseline": b_metrics,
                "lstm": l_metrics,
            }
        except Exception:
            pass

    return {
        "job_id": job_id,
        "filename": os.path.basename(csv_path),
        "status": "done",
        "total_windows": len(window_predictions),
        "windows": window_predictions,
        "latest_attention": latest_attention,
        "latest_top_features": latest_top_features,
        "summary": {
            "max_infiltration_probability": round(float(max_prob), 4),
            "dominant_stage": dominant_stage,
            "dominant_stage_metadata": STAGE_METADATA.get(dominant_stage, STAGE_METADATA["Benign"]),
            "num_windows_flagged": flagged_count,
            "total_windows": len(window_predictions),
            "stage_breakdown": stage_counter,
        },
        "comparison_metrics": comparison_metrics,
    }
