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
) -> pd.DataFrame:
    """Load multiple raw files and concatenate into a single DataFrame.
    
    Args:
        raw_dir: Directory containing raw CSV files
        split: 'train' (Mon-Wed), 'test' (Thu-Fri), or None (all files)
        sample_per_file: Optional max rows to sample per file (for CPU feasibility)
        selected_features: Features to preserve and clean
    """
    all_files = find_raw_files(raw_dir)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    target_files = []
    if split in ("train", "test"):
        allowed_names = DAY_CATEGORIES[split]
        for f in all_files:
            if any(name.lower() in os.path.basename(f).lower() for name in allowed_names):
                target_files.append(f)
    else:
        target_files = all_files

    if not target_files:
        target_files = all_files

    dfs = []
    for f in target_files:
        basename = os.path.basename(f)
        try:
            if sample_per_file is not None:
                # Balanced sampling to ensure attack patterns are captured
                max_b = sample_per_file // 2
                max_a = sample_per_file // 2
                benign_chunks = []
                attack_chunks = []
                
                for chunk in pd.read_csv(
                    f,
                    chunksize=25000,
                    low_memory=False,
                    encoding="utf-8",
                    encoding_errors="replace",
                ):
                    lbl_col = [c for c in chunk.columns if "label" in c.lower()][0]
                    b = chunk[chunk[lbl_col].astype(str).str.strip() == "BENIGN"]
                    a = chunk[chunk[lbl_col].astype(str).str.strip() != "BENIGN"]
                    
                    curr_b = sum(len(x) for x in benign_chunks)
                    if len(b) > 0 and curr_b < max_b:
                        benign_chunks.append(b.head(max_b - curr_b))
                        
                    curr_a = sum(len(x) for x in attack_chunks)
                    if len(a) > 0 and curr_a < max_a:
                        attack_chunks.append(a.head(max_a - curr_a))
                        
                    if sum(len(x) for x in benign_chunks) >= max_b and sum(len(x) for x in attack_chunks) >= max_a:
                        break
                        
                combined_slices = benign_chunks + attack_chunks
                if combined_slices:
                    df = pd.concat(combined_slices).sort_index()
                else:
                    df = pd.read_csv(f, nrows=sample_per_file, low_memory=False, encoding="utf-8", encoding_errors="replace")
            else:
                df = pd.read_csv(f, low_memory=False, encoding="utf-8", encoding_errors="replace")

            df = clean_dataframe(df, selected_features=selected_features)
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
