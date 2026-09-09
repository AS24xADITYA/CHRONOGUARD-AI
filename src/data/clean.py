"""Data cleaning utilities for CIC-IDS-2017 flow data.

Handles known data anomalies in CIC-IDS-2017:
- Leading/trailing whitespace in column names (e.g. ' Destination Port')
- Infinite values in 'Flow Bytes/s' and 'Flow Packets/s'
- Missing or NaN values
- Duplicate flows
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from DataFrame column names."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def clean_inf_and_nan(df: pd.DataFrame, subset_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Replace positive/negative infinity with NaN and drop missing rows.
    
    A known issue in CIC-IDS-2017 is zero-duration flows causing divide-by-zero
    in 'Flow Bytes/s' and 'Flow Packets/s'.
    """
    df = df.copy()
    # Replace infinite representations
    df = df.replace([np.inf, -np.inf], np.nan)
    
    if subset_cols:
        cols_to_check = [c for c in subset_cols if c in df.columns]
        df = df.dropna(subset=cols_to_check)
    else:
        df = df.dropna()
        
    return df


def drop_duplicates_and_constants(df: pd.DataFrame, drop_constants: bool = False) -> pd.DataFrame:
    """Drop exact duplicate rows and optionally columns with zero variance."""
    df = df.copy()
    df = df.drop_duplicates()
    
    if drop_constants:
        # Keep columns that have more than 1 unique value or are non-numeric
        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        # Do not drop label if it happens to be single-class in a small slice
        constant_cols = [c for c in constant_cols if c.lower() != 'label']
        if constant_cols:
            df = df.drop(columns=constant_cols)
            
    return df


def clean_dataframe(df: pd.DataFrame, selected_features: Optional[List[str]] = None) -> pd.DataFrame:
    """End-to-end cleaning of a flow DataFrame.
    
    1. Normalizes column headers.
    2. Filters/replaces infinite and missing values.
    3. Removes duplicate flows.
    """
    df = clean_column_names(df)
    
    # Ensure numeric columns are properly typed
    cols_to_check = selected_features if selected_features else []
    for col in cols_to_check:
        if col in df.columns and col.lower() != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df = clean_inf_and_nan(df, subset_cols=cols_to_check)
    df = drop_duplicates_and_constants(df, drop_constants=False)
    
    return df.reset_index(drop=True)
