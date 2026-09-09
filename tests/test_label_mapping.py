"""Unit tests for MITRE ATT&CK label mapping."""

import pytest
from src.data.label_mapping import (
    STAGE_NAMES,
    STAGE_TO_ID,
    ID_TO_STAGE,
    RAW_LABEL_TO_STAGE,
    STAGE_METADATA,
    map_label_to_stage,
    get_stage_id,
    is_attack_stage,
)


def test_stage_definitions():
    """Verify 6-class canonical stages are well-defined."""
    assert len(STAGE_NAMES) == 6
    assert STAGE_NAMES[0] == "Benign"
    assert "Reconnaissance" in STAGE_NAMES
    assert "Credential Access" in STAGE_NAMES
    assert "Initial Access" in STAGE_NAMES
    assert "Lateral Movement" in STAGE_NAMES
    assert "Impact" in STAGE_NAMES


def test_stage_id_bidirectional():
    """Verify stage name to ID and ID to stage name mapping is bidirectional."""
    for name in STAGE_NAMES:
        stage_id = STAGE_TO_ID[name]
        assert ID_TO_STAGE[stage_id] == name
        assert get_stage_id(name) == stage_id


def test_raw_label_coverage():
    """Verify every raw CIC-IDS-2017 label maps to a valid stage."""
    raw_labels = [
        "BENIGN",
        "PortScan",
        "FTP-Patator",
        "SSH-Patator",
        "Heartbleed",
        "Web Attack – Brute Force",
        "Web Attack - Brute Force",
        "Web Attack – XSS",
        "Web Attack - SQL Injection",
        "Infiltration",
        "Bot",
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
        "DDoS",
    ]

    for raw in raw_labels:
        stage = map_label_to_stage(raw)
        assert stage in STAGE_NAMES, f"Label '{raw}' mapped to invalid stage '{stage}'"


def test_attack_detection():
    """Verify attack vs benign discriminator."""
    assert not is_attack_stage("Benign")
    assert is_attack_stage("Reconnaissance")
    assert is_attack_stage("Credential Access")
    assert is_attack_stage("Initial Access")
    assert is_attack_stage("Lateral Movement")
    assert is_attack_stage("Impact")


def test_stage_metadata():
    """Verify all stages have presentation metadata."""
    for stage in STAGE_NAMES:
        assert stage in STAGE_METADATA
        meta = STAGE_METADATA[stage]
        assert "color" in meta
        assert "description" in meta
        assert "badge_class" in meta
