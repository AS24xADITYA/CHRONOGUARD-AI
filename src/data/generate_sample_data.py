"""Sample dataset generator / extractor for instant web UI demonstrations.

Extracts realistic slices from data/raw/ matching the CICFlowMeter 78-column schema:
1. data/samples/sample_benign.csv (Baseline safe web operations)
2. data/samples/sample_reconnaissance.csv (Transition from benign baseline to active port scan)
3. data/samples/sample_multi_stage_attack.csv (Progression: Benign -> PortScan -> Patator -> DDoS)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.clean import clean_dataframe


def extract_samples_from_raw(
    raw_dir: str = "data/raw",
    output_dir: str = "data/samples",
) -> None:
    """Extract representative sample files from raw dataset CSVs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    monday_file = os.path.join(raw_dir, "Monday-WorkingHours.pcap_ISCX.csv")
    portscan_file = os.path.join(raw_dir, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    tuesday_file = os.path.join(raw_dir, "Tuesday-WorkingHours.pcap_ISCX.csv")
    ddos_file = os.path.join(raw_dir, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

    # 1. Benign Sample (400 pure benign flows)
    if os.path.exists(monday_file):
        print(f"Extracting sample_benign.csv from {monday_file}...")
        df_benign = pd.read_csv(monday_file, nrows=400, low_memory=False, encoding="utf-8", encoding_errors="replace")
        df_benign = clean_dataframe(df_benign)
        df_benign.to_csv(os.path.join(output_dir, "sample_benign.csv"), index=False)
        print(f"  -> Saved {len(df_benign)} flows to {output_dir}/sample_benign.csv")

    # 2. Reconnaissance (PortScan) Sample (100 benign flows followed by 300 port scan flows)
    if os.path.exists(portscan_file) and os.path.exists(monday_file):
        print(f"Extracting sample_reconnaissance.csv...")
        df_m = pd.read_csv(monday_file, nrows=100, low_memory=False, encoding="utf-8", encoding_errors="replace")
        df_m_clean = clean_dataframe(df_m)
        
        # Dense PortScan flows occur around row 97,400
        df_p = pd.read_csv(portscan_file, skiprows=range(1, 97400), nrows=400, low_memory=False, encoding="utf-8", encoding_errors="replace")
        df_p_clean = clean_dataframe(df_p)
        lbl_p = [c for c in df_p_clean.columns if c.lower() == 'label'][0]
        scans = df_p_clean[df_p_clean[lbl_p].astype(str).str.strip() == "PortScan"].head(300)
        
        recon_sample = pd.concat([df_m_clean, scans], ignore_index=True)
        recon_sample.to_csv(os.path.join(output_dir, "sample_reconnaissance.csv"), index=False)
        print(f"  -> Saved {len(recon_sample)} flows to {output_dir}/sample_reconnaissance.csv (Labels: {recon_sample[lbl_p].value_counts().to_dict()})")

    # 3. Multi-Stage Attack Sample (Benign -> PortScan -> Patator -> DDoS)
    print("Extracting sample_multi_stage_attack.csv...")
    stages = []
    
    # Phase 1: Benign baseline (100 flows)
    if os.path.exists(monday_file):
        df_m = pd.read_csv(monday_file, nrows=100, low_memory=False, encoding="utf-8", encoding_errors="replace")
        stages.append(clean_dataframe(df_m))
        
    # Phase 2: Reconnaissance / PortScan (150 flows from row 97,400)
    if os.path.exists(portscan_file):
        df_p = pd.read_csv(portscan_file, skiprows=range(1, 97400), nrows=200, low_memory=False, encoding="utf-8", encoding_errors="replace")
        df_p_clean = clean_dataframe(df_p)
        lbl_p = [c for c in df_p_clean.columns if c.lower() == 'label'][0]
        scans = df_p_clean[df_p_clean[lbl_p].astype(str).str.strip() == "PortScan"].head(150)
        stages.append(scans)

    # Phase 3: Credential Access / Patator (150 flows from row 11,350)
    if os.path.exists(tuesday_file):
        df_t = pd.read_csv(tuesday_file, skiprows=range(1, 11350), nrows=200, low_memory=False, encoding="utf-8", encoding_errors="replace")
        df_t_clean = clean_dataframe(df_t)
        lbl_t = [c for c in df_t_clean.columns if c.lower() == 'label'][0]
        pats = df_t_clean[df_t_clean[lbl_t].astype(str).str.strip().str.contains("Patator", case=False, na=False)].head(150)
        stages.append(pats)
            
    # Phase 4: Impact / DDoS (150 flows from row 18,900)
    if os.path.exists(ddos_file):
        df_d = pd.read_csv(ddos_file, skiprows=range(1, 18900), nrows=200, low_memory=False, encoding="utf-8", encoding_errors="replace")
        df_d_clean = clean_dataframe(df_d)
        lbl_d = [c for c in df_d_clean.columns if c.lower() == 'label'][0]
        ddoses = df_d_clean[df_d_clean[lbl_d].astype(str).str.strip() == "DDoS"].head(150)
        stages.append(ddoses)

    if stages:
        multi_sample = pd.concat(stages, ignore_index=True)
        multi_sample.to_csv(os.path.join(output_dir, "sample_multi_stage_attack.csv"), index=False)
        lbl_m = [c for c in multi_sample.columns if c.lower() == 'label'][0]
        print(f"  -> Saved {len(multi_sample)} flows to {output_dir}/sample_multi_stage_attack.csv (Labels: {multi_sample[lbl_m].value_counts().to_dict()})")


if __name__ == "__main__":
    extract_samples_from_raw()
