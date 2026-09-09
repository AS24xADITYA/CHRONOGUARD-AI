"""Explainability module for ChronoGuard.

Implements two-tier explainability per AI-INSTRUCTIONS/05-ml-algorithms.md Section 6:
1. Primary (live, lightweight): Attention weights from self-attention pooling layer
   showing temporal focus, paired with top driver features within the peak window.
2. Secondary (offline validation): SHAP LinearExplainer on the baseline model.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import os


def humanize_feature_name(feature_name: str) -> str:
    """Format raw feature names into clean, readable labels for SOC analysts."""
    name = feature_name.replace("_mean", " (Mean)").replace("_std", " (Std)")
    name = name.replace("flow_count", "Flow Volume")
    name = name.replace("Init_Win_bytes_forward", "Initial Forward TCP Window")
    name = name.replace("Init_Win_bytes_backward", "Initial Backward TCP Window")
    name = name.replace("Flow IAT", "Inter-Arrival Time")
    return name


def explain_prediction_attention(
    attention_weights: np.ndarray,
    sequence_features: np.ndarray,
    feature_names: List[str],
    top_k_features: int = 4,
) -> Dict[str, Any]:
    """Generate explainability payload from LSTM self-attention weights.
    
    Args:
        attention_weights: Array of shape (seq_len,) summing to 1.0
        sequence_features: Array of shape (seq_len, num_features)
        feature_names: List of feature names corresponding to feature dimension
        top_k_features: Number of top features to report
        
    Returns:
        Dictionary with per-window attention distribution and ranked driving features.
    """
    seq_len = len(attention_weights)
    peak_idx = int(np.argmax(attention_weights))
    peak_weight = float(attention_weights[peak_idx])

    # Formatted attention timeline
    window_attention = []
    for idx, weight in enumerate(attention_weights):
        window_attention.append({
            "relative_window": idx - seq_len + 1,
            "window_offset": idx,
            "attention_weight": round(float(weight), 4),
            "is_peak": (idx == peak_idx),
        })

    # Find dominant features within the peak window
    peak_window_features = sequence_features[peak_idx]
    # Use absolute magnitude of features in standard-scaled space as importance proxy
    abs_magnitudes = np.abs(peak_window_features)
    top_indices = np.argsort(abs_magnitudes)[::-1][:top_k_features]
    
    total_mag = float(np.sum(abs_magnitudes[top_indices])) if np.sum(abs_magnitudes[top_indices]) > 0 else 1.0

    top_features = []
    for rank, idx in enumerate(top_indices, start=1):
        feat_raw_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        feat_val = float(peak_window_features[idx])
        feat_weight = round(float(abs_magnitudes[idx]) / total_mag, 3)
        
        top_features.append({
            "rank": rank,
            "feature": feat_raw_name,
            "display_name": humanize_feature_name(feat_raw_name),
            "value": round(feat_val, 4),
            "weight": feat_weight,
            "impact": "elevated" if feat_val > 0 else "suppressed",
        })

    return {
        "peak_window_offset": peak_idx,
        "peak_attention_weight": round(peak_weight, 4),
        "window_attention": window_attention,
        "top_features": top_features,
    }


def run_offline_shap_analysis(
    baseline_model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_path: str = "results/shap_summary.png",
    max_samples: int = 300,
) -> bool:
    """Run offline SHAP LinearExplainer on baseline model and save summary plot."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        sample_size = min(len(X_sample), max_samples)
        X_sub = X_sample[:sample_size]
        
        explainer = shap.LinearExplainer(baseline_model, X_sub)
        shap_values = explainer.shap_values(X_sub)

        # Plot summary
        plt.figure(figsize=(10, 6))
        clean_names = [humanize_feature_name(f) for f in feature_names[:X_sub.shape[1]]]
        shap.summary_plot(shap_values, X_sub, feature_names=clean_names, show=False, max_display=15)
        plt.title("ChronoGuard Baseline Feature Attribution (SHAP)", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"SHAP summary plot saved to {output_path}")
        return True
    except Exception as e:
        print(f"Note: Offline SHAP generation skipped: {e}")
        return False
