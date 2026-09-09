"""SQLAlchemy data models for ChronoGuard analysis jobs and predictions.

Implements database schema strictly as specified in AI-INSTRUCTIONS/04-database-schema.md.
"""

from datetime import datetime
from typing import Dict, Any, List
import json
from src.db import db


class AnalysisJob(db.Model):
    """Represents a single flow CSV upload and analysis job."""
    __tablename__ = "analysis_job"

    id = db.Column(db.String(36), primary_key=True)          # uuid4 string
    original_filename = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="queued")       # queued|running|done|error
    total_windows = db.Column(db.Integer, default=0)
    overall_max_infiltration_prob = db.Column(db.Float, nullable=True)
    dominant_stage = db.Column(db.String(50), nullable=True)
    error_message = db.Column(db.Text, nullable=True)

    predictions = db.relationship(
        "WindowPrediction",
        backref="job",
        cascade="all, delete-orphan",
        order_by="WindowPrediction.window_index",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to serializable dictionary."""
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "uploaded_at": self.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if self.uploaded_at else None,
            "status": self.status,
            "total_windows": self.total_windows,
            "overall_max_infiltration_prob": round(self.overall_max_infiltration_prob, 4) if self.overall_max_infiltration_prob is not None else 0.0,
            "dominant_stage": self.dominant_stage or "Benign",
            "error_message": self.error_message,
        }


class WindowPrediction(db.Model):
    """Represents the temporal forecast and explainability for a single window."""
    __tablename__ = "window_prediction"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    job_id = db.Column(db.String(36), db.ForeignKey("analysis_job.id"), nullable=False)
    window_index = db.Column(db.Integer, nullable=False)
    window_start = db.Column(db.String(50), nullable=True)
    window_end = db.Column(db.String(50), nullable=True)
    infiltration_probability = db.Column(db.Float, nullable=False)
    predicted_stage = db.Column(db.String(50), nullable=False)
    top_features_json = db.Column(db.Text, nullable=True)   # JSON list of {feature, weight, display_name}
    baseline_probability = db.Column(db.Float, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert window prediction to serializable dictionary."""
        top_feats = []
        if self.top_features_json:
            try:
                top_feats = json.loads(self.top_features_json)
            except Exception:
                top_feats = []

        return {
            "id": self.id,
            "job_id": self.job_id,
            "window_index": self.window_index,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "infiltration_probability": round(self.infiltration_probability, 4),
            "predicted_stage": self.predicted_stage,
            "baseline_probability": round(self.baseline_probability, 4) if self.baseline_probability is not None else 0.0,
            "top_features": top_feats,
        }
