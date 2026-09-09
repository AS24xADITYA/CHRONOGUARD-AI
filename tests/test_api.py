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
    """Verify history audit log loads successfully."""
    res = client.get("/history")
    assert res.status_code == 200
    assert b"Historical Analysis Logs" in res.data


def test_unknown_job_status(client):
    """Verify polling unknown job ID returns 404."""
    res = client.get("/status/non-existent-uuid-12345")
    assert res.status_code == 404
