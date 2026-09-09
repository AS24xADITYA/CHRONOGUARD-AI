"""Sequence windowing and feature aggregation engine for ChronoGuard.

Transforms discrete, uncoordinated network flows into a coherent time series:
1. Identifies flow sessions (via Source IP, Destination Port, or chronological flow chunks)
2. Buckets flows into fixed-size temporal or volume windows
3. Aggregates statistical summaries (mean, std, flow count)
4. Maps majority-vote MITRE ATT&CK labels
5. Constructs overlapping sliding sequence tensors (W=10) with multi-step forecast horizons (K=3)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler

from src.data.label_mapping import map_label_to_stage, get_stage_id, is_attack_stage


def get_feature_names(selected_features: List[str]) -> List[str]:
    """Generate the ordered list of aggregated feature names (mean, std, flow_count)."""
    feature_names = []
    for feat in selected_features:
        feature_names.append(f"{feat}_mean")
        feature_names.append(f"{feat}_std")
    feature_names.append("flow_count")
    return feature_names


def aggregate_single_window(
    window_df: pd.DataFrame,
    selected_features: List[str],
    window_index: int,
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None,
) -> Dict[str, Any]:
    """Aggregate a subset of flows in a window into a single feature vector and label."""
    num_flows = len(window_df)
    feature_values = []
    
    for feat in selected_features:
        if feat in window_df.columns:
            vals = window_df[feat].to_numpy(dtype=np.float32)
            f_mean = float(np.nanmean(vals)) if len(vals) > 0 else 0.0
            f_std = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
            if np.isnan(f_mean):
                f_mean = 0.0
            if np.isnan(f_std):
                f_std = 0.0
        else:
            f_mean = 0.0
            f_std = 0.0
            
        feature_values.append(f_mean)
        feature_values.append(f_std)
        
    feature_values.append(float(num_flows))
    
    # Label extraction via majority vote
    label_col = None
    for col in window_df.columns:
        if col.strip().lower() == "label":
            label_col = col
            break
            
    if label_col is not None and not window_df.empty:
        raw_labels = window_df[label_col].dropna().tolist()
        if raw_labels:
            stages = [map_label_to_stage(lbl) for lbl in raw_labels]
            # Count stage occurrences
            stage_counts = pd.Series(stages).value_counts()
            dominant_stage = stage_counts.index[0]
            # Attack probability: fraction of flows that are non-benign
            attack_flows = sum(1 for s in stages if is_attack_stage(s))
            infiltration_prob = attack_flows / float(len(stages))
        else:
            dominant_stage = "Benign"
            infiltration_prob = 0.0
    else:
        dominant_stage = "Benign"
        infiltration_prob = 0.0

    return {
        "window_index": window_index,
        "window_start": str(window_start) if window_start is not None else None,
        "window_end": str(window_end) if window_end is not None else None,
        "features": np.array(feature_values, dtype=np.float32),
        "dominant_stage": dominant_stage,
        "stage_id": get_stage_id(dominant_stage),
        "infiltration_probability": float(infiltration_prob),
        "flow_count": num_flows,
    }


def bucket_flows_into_windows(
    df: pd.DataFrame,
    selected_features: List[str],
    window_size_flows: int = 20,
    time_col: str = "Timestamp",
    ip_col: str = "Source IP",
) -> List[Dict[str, Any]]:
    """Bucket a cleaned DataFrame of flows into aggregated windows."""
    windows = []
    
    # Check if Source IP is present for session grouping
    has_ip = ip_col in df.columns
    has_time = time_col in df.columns
    
    if has_ip:
        groups = df.groupby(ip_col, sort=False)
    else:
        # Fallback to single contiguous stream or destination port grouping
        groups = [("all", df)]
        
    window_counter = 0
    for _, group_df in groups:
        if has_time:
            try:
                group_df = group_df.sort_values(by=time_col)
            except Exception:
                pass
                
        n_rows = len(group_df)
        if n_rows == 0:
            continue
            
        for start_idx in range(0, n_rows, window_size_flows):
            end_idx = min(start_idx + window_size_flows, n_rows)
            chunk = group_df.iloc[start_idx:end_idx]
            
            w_start = chunk[time_col].iloc[0] if has_time and not chunk.empty else window_counter * 10
            w_end = chunk[time_col].iloc[-1] if has_time and not chunk.empty else (window_counter + 1) * 10
            
            w_data = aggregate_single_window(
                chunk,
                selected_features=selected_features,
                window_index=window_counter,
                window_start=w_start,
                window_end=w_end,
            )
            windows.append(w_data)
            window_counter += 1
            
    return windows


def create_sequences(
    windows: List[Dict[str, Any]],
    sequence_length_w: int = 10,
    forecast_horizon_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Convert a sequence of window dicts into tensors for LSTM training/inference.
    
    Returns:
        X: shape (N, W, num_features)
        y_prob: shape (N, K) - infiltration probability for t+1..t+K
        y_stage: shape (N,) - integer stage class for t+1
        sequence_metadata: list of dicts with window indices for tracking
    """
    total_windows = len(windows)
    min_required = sequence_length_w + forecast_horizon_k
    
    if total_windows < sequence_length_w:
        # If fewer windows than W, pad windows with zeros
        if total_windows == 0:
            return np.empty((0, sequence_length_w, 0)), np.empty((0, forecast_horizon_k)), np.empty((0,)), []
            
        feat_dim = len(windows[0]["features"])
        padded_x = np.zeros((sequence_length_w, feat_dim), dtype=np.float32)
        for i, w in enumerate(windows):
            padded_x[sequence_length_w - len(windows) + i] = w["features"]
            
        X = np.expand_dims(padded_x, axis=0)
        y_prob = np.zeros((1, forecast_horizon_k), dtype=np.float32)
        y_stage = np.array([windows[-1]["stage_id"]], dtype=np.int64)
        meta = [{"window_indices": [w["window_index"] for w in windows]}]
        return X, y_prob, y_stage, meta

    X_list = []
    y_prob_list = []
    y_stage_list = []
    meta_list = []
    
    # Check if we have enough windows for target horizons
    num_sequences = total_windows - sequence_length_w - forecast_horizon_k + 1
    
    if num_sequences <= 0:
        # Fallback for short files: evaluate available sequence steps
        for i in range(total_windows - sequence_length_w + 1):
            x_seq = np.array([w["features"] for w in windows[i:i + sequence_length_w]], dtype=np.float32)
            X_list.append(x_seq)
            # Pad target if needed
            future_windows = windows[i + sequence_length_w:i + sequence_length_w + forecast_horizon_k]
            probs = [w["infiltration_probability"] for w in future_windows]
            while len(probs) < forecast_horizon_k:
                probs.append(probs[-1] if probs else windows[i + sequence_length_w - 1]["infiltration_probability"])
            y_prob_list.append(np.array(probs[:forecast_horizon_k], dtype=np.float32))
            next_stage = future_windows[0]["stage_id"] if future_windows else windows[i + sequence_length_w - 1]["stage_id"]
            y_stage_list.append(next_stage)
            meta_list.append({
                "window_indices": [w["window_index"] for w in windows[i:i + sequence_length_w]],
                "target_window_index": windows[i + sequence_length_w - 1]["window_index"] + 1,
            })
    else:
        for i in range(num_sequences):
            x_seq = np.array([w["features"] for w in windows[i:i + sequence_length_w]], dtype=np.float32)
            targets = windows[i + sequence_length_w:i + sequence_length_w + forecast_horizon_k]
            y_p = np.array([t["infiltration_probability"] for t in targets], dtype=np.float32)
            # Stage target is the immediate next window (t+1)
            y_s = targets[0]["stage_id"]
            
            X_list.append(x_seq)
            y_prob_list.append(y_p)
            y_stage_list.append(y_s)
            meta_list.append({
                "window_indices": [w["window_index"] for w in windows[i:i + sequence_length_w]],
                "target_window_index": targets[0]["window_index"],
            })

    X = np.array(X_list, dtype=np.float32)
    y_prob = np.array(y_prob_list, dtype=np.float32)
    y_stage = np.array(y_stage_list, dtype=np.int64)
    
    return X, y_prob, y_stage, meta_list
