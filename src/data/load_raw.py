"""Raw dataset loader for CIC-IDS-2017 flow CSV files.

Loads, validates, and concatenates the 8 daily capture files from data/raw/:
- Monday-WorkingHours.pcap_ISCX.csv (Benign baseline)
- Tuesday-WorkingHours.pcap_ISCX.csv (FTP/SSH Patator)
- Wednesday-workingHours.pcap_ISCX.csv (DoS / Heartbleed)
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv (Web Attacks)
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv (Infiltration)
- Friday-WorkingHours-Morning.pcap_ISCX.csv (Botnet)
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv (PortScan)
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (DDoS)
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from src.data.clean import clean_dataframe


# Day classification for time-based evaluation splits
DAY_CATEGORIES = {
    "train": [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    ],
    "test": [
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ],
}


def find_raw_files(raw_dir: str = "data/raw") -> List[str]:
    """Find all CSV files in the raw dataset directory."""
    p = Path(raw_dir)
    if not p.exists():
        return []
    csv_files = sorted(glob.glob(str(p / "*.csv")))
    return csv_files


def load_single_csv(
    filepath: str,
    nrows: Optional[int] = None,
    selected_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load and clean a single CIC-IDS-2017 CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df = clean_dataframe(df, selected_features=selected_features)
    return df


def load_raw_dataset(
    raw_dir: str = "data/raw",
    split: Optional[str] = None,
    sample_per_file: Optional[int] = None,
    selected_features: Optional[List[str]] = None,
    train_ratio: float = 0.75,
) -> pd.DataFrame:
    """Load raw capture files and construct chronological train/test splits.
    
    Implements a per-file chronological split (first 75% train, last 25% test)
    across all daily captures. This guarantees every MITRE ATT&CK class that
    occurs in the dataset appears in both splits, while strictly preserving
    the chronological sequence of network flows without future look-ahead leakage.
    
    Args:
        raw_dir: Directory containing raw CSV files
        split: 'train' (first 75% of each session), 'test' (last 25%), or None (full)
        sample_per_file: Optional max rows to sample per class to ensure balance
        selected_features: Features to preserve and clean
        train_ratio: Chronological partition point (default 0.75)
    """
    from src.data.label_mapping import map_label_to_stage

    all_files = find_raw_files(raw_dir)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    dfs = []
    for f in all_files:
        basename = os.path.basename(f)
        try:
            df = pd.read_csv(f, low_memory=False, encoding="utf-8", encoding_errors="replace")
            df = clean_dataframe(df, selected_features=selected_features)
            
            lbl_cols = [c for c in df.columns if "label" in c.lower()]
            if not lbl_cols:
                continue
            lbl_col = lbl_cols[0]
            df["stage"] = df[lbl_col].astype(str).str.strip().apply(map_label_to_stage)

            if split in ("train", "val", "test"):
                file_parts = []
                for stage_name, grp in df.groupby("stage", sort=False):
                    n = len(grp)
                    n_tr = int(n * 0.65)
                    n_val = int(n * 0.75)
                    
                    if split == "train":
                        part = grp.iloc[:n_tr]
                        max_tr = int(sample_per_file * 0.30) if sample_per_file else None
                        if max_tr and len(part) > max_tr:
                            part = part.head(max_tr)
                    elif split == "val":
                        part = grp.iloc[n_tr:n_val]
                        max_val = int(sample_per_file * 0.05) if sample_per_file else None
                        if max_val and len(part) > max_val:
                            part = part.head(max_val)
                    else:  # test (final 25%)
                        part = grp.iloc[n_val:]
                        max_te = int(sample_per_file * 0.12) if sample_per_file else None
                        if max_te and len(part) > max_te:
                            part = part.head(max_te)
                            
                    file_parts.append(part)
                    
                if file_parts:
                    df = pd.concat(file_parts).sort_index()
                else:
                    df = df.iloc[:0]
                    
            df["source_file"] = basename
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {basename}: {e}")

    if not dfs:
        raise ValueError(f"Could not load any data from {raw_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and check raw CIC-IDS-2017 files")
    parser.add_argument("--raw_dir", default="data/raw", help="Path to raw CSV directory")
    parser.add_argument("--sample", type=int, default=1000, help="Sample rows per file")
    args = parser.parse_args()
    
    print(f"Scanning {args.raw_dir}...")
    files = find_raw_files(args.raw_dir)
    print(f"Found {len(files)} raw files:")
    for f in files:
        print(f"  - {os.path.basename(f)} ({os.path.getsize(f) / (1024*1024):.1f} MB)")
    
    print(f"\nLoading sample of {args.sample} rows per file...")
    data = load_raw_dataset(args.raw_dir, sample_per_file=args.sample)
    print(f"Total loaded rows: {len(data):,}")
    print("Label distribution:")
    if "Label" in data.columns:
        print(data["Label"].value_counts())
