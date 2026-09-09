"""MITRE ATT&CK stage mapping for CIC-IDS-2017 flow labels.

Implements the simplified 6-class scheme (5 attack stages + Benign)
strictly as defined in AI-INSTRUCTIONS/05-ml-algorithms.md Section 3.
"""

from typing import Dict, Any

# Canonical MITRE Stage names (6 classes)
STAGE_NAMES = [
    "Benign",
    "Reconnaissance",
    "Credential Access",
    "Initial Access",
    "Lateral Movement",
    "Impact",
]

STAGE_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(STAGE_NAMES)}
ID_TO_STAGE: Dict[int, str] = {idx: name for idx, name in enumerate(STAGE_NAMES)}

# Exact mapping from raw CIC-IDS-2017 labels to mapped stages
RAW_LABEL_TO_STAGE: Dict[str, str] = {
    # Benign
    "benign": "Benign",
    
    # Reconnaissance
    "portscan": "Reconnaissance",
    
    # Credential Access (Brute Force)
    "ftp-patator": "Credential Access",
    "ssh-patator": "Credential Access",
    
    # Initial Access
    "heartbleed": "Initial Access",
    "web attack – brute force": "Initial Access",
    "web attack - brute force": "Initial Access",
    "web attack – xss": "Initial Access",
    "web attack - xss": "Initial Access",
    "web attack – sql injection": "Initial Access",
    "web attack - sql injection": "Initial Access",
    "web attack": "Initial Access",
    
    # Lateral Movement / C2
    "infiltration": "Lateral Movement",
    "bot": "Lateral Movement",
    
    # Impact (Denial of Service)
    "dos hulk": "Impact",
    "dos goldeneye": "Impact",
    "dos slowloris": "Impact",
    "dos slowhttptest": "Impact",
    "ddos": "Impact",
}

# Rich metadata for UI presentation and explanations
STAGE_METADATA: Dict[str, Dict[str, Any]] = {
    "Benign": {
        "id": 0,
        "tactic": "Normal Operations",
        "description": "Baseline network traffic with standard flow characteristics.",
        "color": "#10B981",       # Emerald
        "badge_class": "badge-safe",
        "severity": "None",
        "mitre_id": "N/A",
    },
    "Reconnaissance": {
        "id": 1,
        "tactic": "TA0043 - Reconnaissance",
        "description": "Active port scanning, host discovery, or service probing.",
        "color": "#F59E0B",       # Amber
        "badge_class": "badge-warning",
        "severity": "Low-Medium",
        "mitre_id": "TA0043",
    },
    "Credential Access": {
        "id": 2,
        "tactic": "TA0006 - Credential Access",
        "description": "Automated brute-force or dictionary attempts on authentication services.",
        "color": "#F97316",       # Orange
        "badge_class": "badge-warning",
        "severity": "Medium-High",
        "mitre_id": "TA0006",
    },
    "Initial Access": {
        "id": 3,
        "tactic": "TA0001 - Initial Access",
        "description": "Application exploits (SQLi, XSS) or vulnerability probes attempting entry.",
        "color": "#EF4444",       # Red
        "badge_class": "badge-danger",
        "severity": "High",
        "mitre_id": "TA0001",
    },
    "Lateral Movement": {
        "id": 4,
        "tactic": "TA0008 - Lateral Movement",
        "description": "Host infiltration, internal pivoting, or automated bot activity.",
        "color": "#DC2626",       # Deep Red
        "badge_class": "badge-danger",
        "severity": "Critical",
        "mitre_id": "TA0008",
    },
    "Impact": {
        "id": 5,
        "tactic": "TA0040 - Impact",
        "description": "Denial-of-Service resource exhaustion targeting service availability.",
        "color": "#B91C1C",       # Dark Red
        "badge_class": "badge-critical",
        "severity": "Critical",
        "mitre_id": "TA0040",
    },
}


def normalize_label_str(label: Any) -> str:
    """Normalize a raw label string for lookup."""
    if not isinstance(label, str):
        label = str(label)
    # Strip whitespace, convert to lower, normalize unicode en-dash to hyphen
    return label.strip().lower().replace("–", "-")


def map_label_to_stage(raw_label: str) -> str:
    """Map a raw CIC-IDS-2017 label to its canonical MITRE ATT&CK stage.
    
    Defaults to 'Benign' if unrecognized.
    """
    normalized = normalize_label_str(raw_label)
    if normalized in RAW_LABEL_TO_STAGE:
        return RAW_LABEL_TO_STAGE[normalized]
    
    # Partial substring matches for edge cases
    for key, stage in RAW_LABEL_TO_STAGE.items():
        if key in normalized:
            return stage
            
    return "Benign"


def get_stage_id(stage_name: str) -> int:
    """Return integer ID (0-5) for a stage name."""
    return STAGE_TO_ID.get(stage_name, 0)


def is_attack_stage(stage_name: str) -> bool:
    """Return True if stage represents an attack (non-Benign)."""
    return stage_name != "Benign"
