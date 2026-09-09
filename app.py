"""ChronoGuard Flask Application Entrypoint.

Provides the SOC web interface and REST API endpoints:
- GET  /              : Landing page with interactive architecture breakdown
- GET, POST /upload   : CSV file dropzone and sample benchmark launcher
- GET  /status/<id>   : Real-time background job polling API
- GET  /results/<id>  : Interactive SOC dashboard (timeline, attention, metrics)
- GET  /history       : Audit history of analyzed traffic captures
- GET  /api/health    : System health check
"""

import os
import uuid
import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from src.db import db
from src.db.models import AnalysisJob, WindowPrediction
from src.inference.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Application Factory & Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = INSTANCE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static",
)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "chronoguard-dev-secret-key-2026")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{INSTANCE_DIR / 'chronoguard.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit per TRD

db.init_app(app)

# In-memory registry for non-blocking asynchronous jobs
JOB_STATUS_REGISTRY = {}
JOB_RESULTS_CACHE = {}


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------------
# Asynchronous Background Worker
# ---------------------------------------------------------------------------

def process_analysis_job(job_id: str, filepath: str, filename: str, is_sample: bool = False):
    """Execute inference pipeline in background thread and update SQLite."""
    with app.app_context():
        try:
            JOB_STATUS_REGISTRY[job_id] = {"status": "running", "error": None}
            job_record = db.session.get(AnalysisJob, job_id)
            if job_record:
                job_record.status = "running"
                db.session.commit()

            # Execute shared inference pipeline
            results = run_pipeline(csv_path=filepath, job_id=job_id)
            JOB_RESULTS_CACHE[job_id] = results

            # Persist summary to SQLite
            if job_record:
                job_record.status = "done"
                job_record.total_windows = results["total_windows"]
                job_record.overall_max_infiltration_prob = results["summary"]["max_infiltration_probability"]
                job_record.dominant_stage = results["summary"]["dominant_stage"]

                # Persist window predictions
                for w in results["windows"]:
                    wp = WindowPrediction(
                        job_id=job_id,
                        window_index=w["window_index"],
                        window_start=w.get("window_start"),
                        window_end=w.get("window_end"),
                        infiltration_probability=w["infiltration_probability"],
                        predicted_stage=w["predicted_stage"],
                        baseline_probability=w.get("baseline_probability", 0.0),
                        top_features_json=json.dumps(w.get("top_features", [])),
                    )
                    db.session.add(wp)

                db.session.commit()

            JOB_STATUS_REGISTRY[job_id] = {"status": "done", "error": None}

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error in job {job_id}: {error_trace}")
            JOB_STATUS_REGISTRY[job_id] = {"status": "error", "error": str(e)}
            try:
                job_record = db.session.get(AnalysisJob, job_id)
                if job_record:
                    job_record.status = "error"
                    job_record.error_message = str(e)
                    db.session.commit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Application Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page with SOC presentation and architecture breakdown."""
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """File upload dropzone and sample benchmark runner."""
    if request.method == "POST":
        sample_name = request.args.get("sample") or request.form.get("sample")

        # Case 1: Pre-packaged benchmark sample requested
        if sample_name:
            # Sanitize sample filename
            safe_sample = os.path.basename(sample_name)
            sample_path = BASE_DIR / "data" / "samples" / safe_sample
            if not sample_path.exists():
                return jsonify({"error": f"Sample {safe_sample} not found on server."}), 404

            job_id = str(uuid.uuid4())
            job = AnalysisJob(
                id=job_id,
                original_filename=safe_sample,
                status="queued",
            )
            db.session.add(job)
            db.session.commit()

            JOB_STATUS_REGISTRY[job_id] = {"status": "queued", "error": None}
            thread = threading.Thread(
                target=process_analysis_job,
                args=(job_id, str(sample_path), safe_sample, True),
                daemon=True,
            )
            thread.start()
            return jsonify({"job_id": job_id, "status": "queued"}), 202

        # Case 2: Custom CSV file upload
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename provided."}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Invalid file type. Only CSV files are supported."}), 400

        job_id = str(uuid.uuid4())
        safe_name = secure_filename(file.filename) or "traffic_capture.csv"
        dest_filename = f"{job_id}_{safe_name}"
        save_path = UPLOAD_DIR / dest_filename
        file.save(save_path)

        job = AnalysisJob(
            id=job_id,
            original_filename=safe_name,
            status="queued",
        )
        db.session.add(job)
        db.session.commit()

        JOB_STATUS_REGISTRY[job_id] = {"status": "queued", "error": None}
        thread = threading.Thread(
            target=process_analysis_job,
            args=(job_id, str(save_path), safe_name, False),
            daemon=True,
        )
        thread.start()

        return jsonify({"job_id": job_id, "status": "queued"}), 202

    return render_template("upload.html")


@app.route("/status/<job_id>")
def job_status(job_id: str):
    """Poll asynchronous job processing state."""
    # Check in-memory status registry
    if job_id in JOB_STATUS_REGISTRY:
        return jsonify(JOB_STATUS_REGISTRY[job_id])

    # Fallback check against database
    job = db.session.get(AnalysisJob, job_id)
    if job:
        return jsonify({"status": job.status, "error": job.error_message})

    return jsonify({"error": "Job ID not found."}), 404


@app.route("/results/<job_id>")
def results(job_id: str):
    """Render comprehensive SOC Dashboard for an analysis job."""
    job = db.session.get(AnalysisJob, job_id)
    if not job:
        flash(f"Job {job_id} not found.", "danger")
        return redirect(url_for("history"))

    # If cached in-memory, use full enriched results payload
    if job_id in JOB_RESULTS_CACHE:
        results_data = JOB_RESULTS_CACHE[job_id]
    else:
        # Reconstruct from SQLite database
        predictions = WindowPrediction.query.filter_by(job_id=job_id).order_by(WindowPrediction.window_index).all()
        windows_data = [p.to_dict() for p in predictions]

        # Load metrics from disk if available
        comp_metrics = {}
        if (BASE_DIR / "results" / "baseline_metrics.json").exists() and (BASE_DIR / "results" / "lstm_metrics.json").exists():
            try:
                with open(BASE_DIR / "results" / "baseline_metrics.json") as fb:
                    b_metrics = json.load(fb)
                with open(BASE_DIR / "results" / "lstm_metrics.json") as fl:
                    l_metrics = json.load(fl)
                comp_metrics = {"baseline": b_metrics, "lstm": l_metrics}
            except Exception:
                pass

        from src.data.label_mapping import STAGE_METADATA
        dominant_stage = job.dominant_stage or "Benign"

        results_data = {
            "job_id": job.id,
            "filename": job.original_filename,
            "status": job.status,
            "total_windows": len(windows_data),
            "windows": windows_data,
            "latest_attention": windows_data[-1].get("attention_timeline", []) if windows_data else [],
            "latest_top_features": windows_data[-1].get("top_features", []) if windows_data else [],
            "summary": {
                "max_infiltration_probability": job.overall_max_infiltration_prob or 0.0,
                "dominant_stage": dominant_stage,
                "dominant_stage_metadata": STAGE_METADATA.get(dominant_stage, STAGE_METADATA["Benign"]),
                "num_windows_flagged": sum(1 for w in windows_data if w["infiltration_probability"] >= 0.66),
                "total_windows": len(windows_data),
            },
            "comparison_metrics": comp_metrics,
        }

    return render_template("results.html", job=job, results=results_data)


@app.route("/history")
def history():
    """List all previous analysis runs from SQLite."""
    jobs = AnalysisJob.query.order_by(AnalysisJob.uploaded_at.desc()).all()
    return render_template("history.html", jobs=jobs)


@app.route("/api/health")
def health():
    """Simple health check endpoint for monitoring."""
    from datetime import timezone
    return jsonify({
        "status": "healthy",
        "service": "ChronoGuard Forecaster",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "connected",
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle uploads exceeding MAX_CONTENT_LENGTH (50MB)."""
    if request.is_json or request.path.startswith("/upload"):
        return jsonify({
            "error": "File exceeds the 50 MB upload limit. Please upload a smaller flow slice or pre-sampled extract."
        }), 413
    flash("File exceeds the maximum allowed size of 50 MB.", "error")
    return redirect(url_for("upload")), 413


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
