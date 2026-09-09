"""Integration tests for the inference pipeline components."""

import os
import pytest
import pandas as pd
import numpy as np

from src.data.clean import clean_dataframe, clean_column_names, clean_inf_and_nan
from src.data.label_mapping import map_label_to_stage


def test_clean_dataframe():
    """Verify data cleaning on raw flow records with anomalies."""
    raw_df = pd.DataFrame({
        " Destination Port ": [80, 443, 22, 80],
        " Flow Duration ": [1000, 2000, 1500, 1000],
        "Flow Bytes/s": [500.0, np.inf, 120.0, 500.0],  # Has infinity
        "Flow Packets/s": [10.0, 20.0, -np.inf, 10.0],  # Has -infinity
        "Label": ["BENIGN", "PortScan", "BENIGN", "BENIGN"],
    })

    # Test column name cleaning
    cleaned_cols = clean_column_names(raw_df)
    assert "Destination Port" in cleaned_cols.columns
    assert "Flow Duration" in cleaned_cols.columns

    # Test end-to-end cleaning
    cleaned = clean_dataframe(raw_df)
    # The two rows with inf/-inf should be dropped, and the remaining duplicate row dropped
    assert len(cleaned) == 1
    assert not np.isinf(cleaned["Flow Bytes/s"].values).any()
    assert not np.isnan(cleaned["Flow Bytes/s"].values).any()


def test_label_normalization():
    """Verify robust handling of whitespace and case in labels."""
    assert map_label_to_stage("  BENIGN  ") == "Benign"
    assert map_label_to_stage("PortScan") == "Reconnaissance"
    assert map_label_to_stage("FTP-Patator") == "Credential Access"
    assert map_label_to_stage("Web Attack – XSS") == "Initial Access"
    assert map_label_to_stage("Infiltration") == "Lateral Movement"
    assert map_label_to_stage("DDoS") == "Impact"
