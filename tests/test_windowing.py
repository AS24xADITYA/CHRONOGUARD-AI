"""Unit tests for sequence windowing and aggregation engine."""

import numpy as np
import pandas as pd
import pytest

from src.data.windowing import (
    aggregate_single_window,
    bucket_flows_into_windows,
    create_sequences,
    get_feature_names,
)


@pytest.fixture
def sample_flows_df():
    """Create a synthetic DataFrame simulating flow records."""
    np.random.seed(42)
    n_flows = 60
    return pd.DataFrame({
        "Flow Duration": np.random.uniform(100, 5000, n_flows),
        "Total Fwd Packets": np.random.randint(1, 20, n_flows),
        "Total Backward Packets": np.random.randint(1, 20, n_flows),
        "SYN Flag Count": np.random.choice([0, 1], n_flows),
        "Label": ["BENIGN"] * 40 + ["PortScan"] * 20,
        "Timestamp": [f"2017-07-05 09:00:{i:02d}" for i in range(n_flows)],
    })


def test_feature_names():
    """Verify aggregated feature name calculation."""
    features = ["Flow Duration", "Total Fwd Packets"]
    names = get_feature_names(features)
    expected = [
        "Flow Duration_mean",
        "Flow Duration_std",
        "Total Fwd Packets_mean",
        "Total Fwd Packets_std",
        "flow_count",
    ]
    assert names == expected


def test_aggregate_single_window(sample_flows_df):
    """Verify aggregation of a single window of flows."""
    features = ["Flow Duration", "Total Fwd Packets"]
    agg = aggregate_single_window(
        sample_flows_df.iloc[:20],
        selected_features=features,
        window_index=0,
    )

    assert agg["window_index"] == 0
    assert agg["flow_count"] == 20
    assert agg["dominant_stage"] == "Benign"
    assert agg["infiltration_probability"] == 0.0
    assert len(agg["features"]) == 2 * len(features) + 1


def test_bucket_flows_into_windows(sample_flows_df):
    """Verify splitting flows into sequential windows."""
    features = ["Flow Duration", "Total Fwd Packets", "SYN Flag Count"]
    windows = bucket_flows_into_windows(
        sample_flows_df,
        selected_features=features,
        window_size_flows=20,
    )

    # 60 flows / 20 flows per window = 3 windows
    assert len(windows) == 3
    assert windows[0]["dominant_stage"] == "Benign"
    assert windows[1]["dominant_stage"] == "Benign"
    assert windows[2]["dominant_stage"] == "Reconnaissance"
    assert windows[2]["infiltration_probability"] == 1.0


def test_create_sequences_dimensions():
    """Verify sequence sliding window dimensions and horizon alignment."""
    # Create 15 synthetic windows
    windows = []
    for i in range(15):
        windows.append({
            "window_index": i,
            "features": np.random.randn(9).astype(np.float32),
            "dominant_stage": "Benign" if i < 10 else "Reconnaissance",
            "stage_id": 0 if i < 10 else 1,
            "infiltration_probability": 0.0 if i < 10 else 1.0,
            "flow_count": 20,
        })

    seq_w = 10
    forecast_k = 3
    X, y_prob, y_stage, meta = create_sequences(windows, sequence_length_w=seq_w, forecast_horizon_k=forecast_k)

    # 15 windows - 10 seq_w - 3 forecast_k + 1 = 3 sequences
    expected_sequences = 15 - 10 - 3 + 1
    assert X.shape == (expected_sequences, seq_w, 9)
    assert y_prob.shape == (expected_sequences, forecast_k)
    assert y_stage.shape == (expected_sequences,)

    # Verify no lookahead in metadata
    for idx, m in enumerate(meta):
        assert m["window_indices"] == list(range(idx, idx + seq_w))
        assert m["target_window_index"] == idx + seq_w
