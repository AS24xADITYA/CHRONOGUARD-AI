"""Integration smoke tests for Flask web endpoints."""

import pytest
from app import app, db


@pytest.fixture
def client():
    """Create Flask test client with test configuration."""
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client


def test_health_endpoint(client):
    """Verify system health check endpoint."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_landing_page(client):
    """Verify landing page loads successfully."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"CHRONO" in res.data
    assert b"GUARD" in res.data


def test_upload_page(client):
    """Verify upload page loads successfully."""
    res = client.get("/upload")
    assert res.status_code == 200
    assert b"dropzone" in res.data or b"file-input" in res.data


def test_history_page(client):
    """Verify history audit log loads successfully when empty."""
    res = client.get("/history")
    assert res.status_code == 200
    assert b"Historical Analysis Logs" in res.data


def test_history_page_with_records(client):
    """Verify history audit log loads successfully with real job records."""
    from src.db.models import AnalysisJob
    with app.app_context():
        job = AnalysisJob(
            id="test-job-uuid-1234",
            original_filename="sample_capture.csv",
            status="done",
            total_windows=10,
            overall_max_infiltration_prob=0.85,
            dominant_stage="Reconnaissance",
        )
        db.session.add(job)
        db.session.commit()

    res = client.get("/history")
    assert res.status_code == 200
    assert b"sample_capture.csv" in res.data
    assert b"Reconnaissance" in res.data
    assert b"85.0%" in res.data


def test_unknown_job_status(client):
    """Verify polling unknown job ID returns 404."""
    res = client.get("/status/non-existent-uuid-12345")
    assert res.status_code == 404
