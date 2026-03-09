import uuid
import sqlite3
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from detection import process_video

# Configuration and Paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed_videos"
DATABASE_PATH = BASE_DIR / "urban_safety.db"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["PROCESSED_FOLDER"] = str(PROCESSED_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# Ensure required directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)

# Database initialization helper
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS safety_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT,
            latitude REAL,
            longitude REAL,
            near_miss_count INTEGER,
            pedestrian_conflict_count INTEGER DEFAULT 0,
            sudden_brake_count INTEGER DEFAULT 0,
            rash_event_count INTEGER DEFAULT 0,
            video_risk_score REAL DEFAULT 0,
            time_bucket TEXT DEFAULT 'Day',
            time_risk_level TEXT DEFAULT 'Low',
            risk_level TEXT,
            risk_score REAL,
            video_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Backward-compatible schema updates for existing databases.
    for alter in (
        "ALTER TABLE safety_reports ADD COLUMN pedestrian_conflict_count INTEGER DEFAULT 0",
        "ALTER TABLE safety_reports ADD COLUMN sudden_brake_count INTEGER DEFAULT 0",
        "ALTER TABLE safety_reports ADD COLUMN rash_event_count INTEGER DEFAULT 0",
        "ALTER TABLE safety_reports ADD COLUMN video_risk_score REAL DEFAULT 0",
        "ALTER TABLE safety_reports ADD COLUMN time_bucket TEXT DEFAULT 'Day'",
        "ALTER TABLE safety_reports ADD COLUMN time_risk_level TEXT DEFAULT 'Low'",
    ):
        try:
            cursor.execute(alter)
        except sqlite3.OperationalError:
            pass
    # Backfill older rows that predate video_risk_score with a conservative estimate.
    cursor.execute(
        """
        UPDATE safety_reports
        SET video_risk_score = (near_miss_count * 2)
        WHERE (video_risk_score IS NULL OR video_risk_score = 0)
          AND near_miss_count > 0
        """
    )
    # Backfill time_bucket from stored timestamp where missing.
    cursor.execute(
        """
        UPDATE safety_reports
        SET time_bucket = CASE
            WHEN CAST(strftime('%H', timestamp) AS INTEGER) >= 6 AND CAST(strftime('%H', timestamp) AS INTEGER) < 10 THEN 'Morning'
            WHEN CAST(strftime('%H', timestamp) AS INTEGER) >= 10 AND CAST(strftime('%H', timestamp) AS INTEGER) < 16 THEN 'Day'
            WHEN CAST(strftime('%H', timestamp) AS INTEGER) >= 16 AND CAST(strftime('%H', timestamp) AS INTEGER) < 20 THEN 'Evening'
            ELSE 'Night'
        END
        WHERE time_bucket IS NULL OR time_bucket = ''
        """
    )
    # Backfill time_risk_level from per-video near miss counts where missing.
    cursor.execute(
        """
        UPDATE safety_reports
        SET time_risk_level = CASE
            WHEN near_miss_count <= 2 THEN 'Low'
            WHEN near_miss_count <= 5 THEN 'Medium'
            ELSE 'High'
        END
        WHERE time_risk_level IS NULL OR time_risk_level = ''
        """
    )
    # Keep persisted per-video levels aligned with current detector calibration.
    cursor.execute(
        """
        UPDATE safety_reports
        SET risk_level = CASE
            WHEN COALESCE(risk_score, 0) >= 30 THEN 'HIGH'
            WHEN COALESCE(risk_score, 0) >= 12 THEN 'MEDIUM'
            ELSE 'LOW'
        END
        WHERE risk_score IS NOT NULL
        """
    )
    conn.commit()
    conn.close()

init_db()


def compute_video_risk_metrics(result: dict) -> dict:
    events = result.get("events", []) or []
    pedestrian_conflicts = 0
    for e in events:
        obj_a = str(e.get("object_a", "")).strip().lower()
        obj_b = str(e.get("object_b", "")).strip().lower()
        if obj_a == "pedestrian" or obj_b == "pedestrian":
            pedestrian_conflicts += 1

    near_miss_count = int(result.get("near_miss_count", len(events)) or 0)
    non_ped_near_miss = max(0, near_miss_count - pedestrian_conflicts)
    sudden_brake_count = int(result.get("sudden_brake_count", 0) or 0)
    rash_event_count = int(result.get("rash_event_count", 0) or 0)

    # Event severity scoring (user policy):
    # near miss = 2, pedestrian conflict = 3
    # sudden brake = 1
    # rash events are treated as sudden-brake proxy for hackathon prototype unless
    # a dedicated sudden_brake_count is provided by detector output.
    effective_sudden_brake_count = sudden_brake_count if sudden_brake_count > 0 else rash_event_count
    video_risk_score = (2 * non_ped_near_miss) + (3 * pedestrian_conflicts) + (1 * effective_sudden_brake_count)
    return {
        "pedestrian_conflict_count": pedestrian_conflicts,
        "sudden_brake_count": effective_sudden_brake_count,
        "rash_event_count": rash_event_count,
        "video_risk_score": float(video_risk_score),
    }


# Save processed results to the persistent database
def save_report_to_db(result, video_filename, incident_time_str=""):
    metrics = compute_video_risk_metrics(result)
    time_bucket = ""
    if incident_time_str:
        try:
            parsed = datetime.strptime(incident_time_str.strip(), "%H:%M")
            time_bucket = get_time_bucket(parsed)
        except ValueError:
            time_bucket = ""
    if not time_bucket:
        now = datetime.now()
        time_bucket = get_time_bucket(now)
    time_risk_level = classify_time_risk_from_near_miss(float(result.get("near_miss_count", 0) or 0))
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO safety_reports 
        (location_name, latitude, longitude, near_miss_count, pedestrian_conflict_count, sudden_brake_count, rash_event_count, video_risk_score, time_bucket, time_risk_level, risk_level, risk_score, video_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result.get('location_name', 'Unknown'),
        result.get('latitude'),
        result.get('longitude'),
        result.get('near_miss_count'),
        metrics["pedestrian_conflict_count"],
        metrics["sudden_brake_count"],
        metrics["rash_event_count"],
        metrics["video_risk_score"],
        time_bucket,
        time_risk_level,
        result.get('risk_level'),
        result.get('risk_score'), # Supported by detection.py
        video_filename
    ))
    conn.commit()
    conn.close()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_time_bucket(dt: datetime) -> str:
    hour = dt.hour
    if 6 <= hour < 10:
        return "Morning"
    if 10 <= hour < 16:
        return "Day"
    if 16 <= hour < 20:
        return "Evening"
    return "Night"


def classify_time_risk_from_near_miss(value: float) -> str:
    if value <= 2:
        return "Low"
    if value <= 5:
        return "Medium"
    return "High"


def classify_location_status(number_of_videos: int, location_risk_score):
    if number_of_videos < 3:
        return "Insufficient Data", "#94A3B8"
    if location_risk_score is None:
        return "Insufficient Data", "#94A3B8"
    if 0.0 <= location_risk_score < 1.0:
        return "Safe", "#26D07C"
    if 1.0 <= location_risk_score < 3.0:
        return "Moderate Risk", "#FF9800"
    if 3.0 <= location_risk_score < 5.0:
        return "High Risk", "#FF2E63"
    return "Dangerous", "#B10F2E"


def to_display_risk_level(status: str) -> tuple[str, str]:
    # Normalize multiple backend/UI labels into consistent 3-tier map labels.
    token = str(status or "").strip().lower().replace("_", " ").replace("-", " ")
    token = " ".join(token.split())

    high_tokens = {"high", "high risk", "dangerous", "very high", "critical"}
    medium_tokens = {"medium", "moderate", "moderate risk", "medium risk"}
    low_tokens = {"low", "safe", "low risk"}

    if token in high_tokens:
        return "High", "#FF2E63"
    if token in medium_tokens:
        return "Medium", "#FF9800"
    if token in low_tokens:
        return "Low", "#26D07C"
    return "Insufficient Data", "#94A3B8"


def fetch_time_bucket_zones(selected_bucket=None):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    params = []
    where_sql = ""
    if selected_bucket:
        where_sql = "WHERE time_bucket = ?"
        params.append(selected_bucket)

    query = f"""
        SELECT
            ROUND(latitude, 3) AS zone_lat,
            ROUND(longitude, 3) AS zone_lng,
            MAX(location_name) AS location_name,
            time_bucket,
            COUNT(*) AS number_of_videos,
            AVG(near_miss_count) AS avg_near_miss_count,
            SUM(video_risk_score) AS total_video_risk_score,
            SUM(sudden_brake_count) AS total_sudden_brakes,
            SUM(pedestrian_conflict_count) AS total_pedestrian_conflicts
        FROM safety_reports
        {where_sql}
        GROUP BY zone_lat, zone_lng, time_bucket
        ORDER BY number_of_videos DESC
    """
    rows = cursor.execute(query, params).fetchall()
    conn.close()

    zones = []
    for row in rows:
        z = dict(row)
        number_of_videos = int(z.get("number_of_videos", 0) or 0)
        total_video_risk_score = float(z.get("total_video_risk_score", 0.0) or 0.0)
        location_risk_score = None
        if number_of_videos >= 3:
            location_risk_score = total_video_risk_score / number_of_videos

        raw_status, _raw_color = classify_location_status(number_of_videos, location_risk_score)
        display_status, display_color = to_display_risk_level(raw_status)
        z["raw_risk_level"] = raw_status
        z["risk_level"] = display_status
        z["color"] = display_color
        z["avg_near_miss_count"] = round(float(z.get("avg_near_miss_count", 0.0) or 0.0), 2)
        z["location_risk_score"] = round(location_risk_score, 2) if location_risk_score is not None else None
        zones.append(z)
    return zones

@app.route("/")
def index():
    default_bucket = "Morning"
    zones = fetch_time_bucket_zones(default_bucket)
    return render_template("index.html", zones=zones, selected_bucket=default_bucket)


@app.route("/api/time-zones")
def api_time_zones():
    bucket = request.args.get("bucket", type=str, default="").strip()
    valid = {"Morning", "Day", "Evening", "Night"}
    selected = bucket if bucket in valid else None
    zones = fetch_time_bucket_zones(selected)
    return jsonify({"bucket": selected or "All", "zones": zones})

@app.route("/delete/<int:report_id>", methods=["POST"])
def delete_report(report_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT video_path FROM safety_reports WHERE id = ?",
        (report_id,),
    ).fetchone()
    cursor.execute('DELETE FROM safety_reports WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()

    if row and row[0]:
        video_path = PROCESSED_FOLDER / row[0]
        video_path.unlink(missing_ok=True)

    return redirect(url_for('history'))
@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    if request.method == "GET":
        return render_template("upload.html")

    video = request.files.get("video")
    latitude = request.form.get("latitude", type=float)
    longitude = request.form.get("longitude", type=float)
    location_name = request.form.get("location_name", type=str, default="")
    incident_time = request.form.get("incident_time", type=str, default="")

    # Basic Validation
    if not video or video.filename == "":
        return render_template("upload.html", error="Please choose a video file.")

    if not allowed_file(video.filename):
        return render_template("upload.html", error="Unsupported file type. Use MP4/MOV/AVI/MKV.")

    if latitude is None or longitude is None:
        return render_template("upload.html", error="Please pick a location on the map.")

    # File naming and paths
    original_name = secure_filename(video.filename)
    stem = Path(original_name).stem
    ext = Path(original_name).suffix.lower()
    uid = uuid.uuid4().hex[:10]

    upload_name = f"{stem}_{uid}{ext}"
    output_name = f"annotated_{stem}_{uid}.mp4"

    upload_path = UPLOAD_FOLDER / upload_name
    output_path = PROCESSED_FOLDER / output_name

    video.save(upload_path)

    try:
        # process_video triggers AI logic, TTC calculations, and risk scoring
        result = process_video(
            input_video_path=str(upload_path),
            output_video_path=str(output_path),
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
        )
        # Add to zone risk map/history only when the clip is zone-risk eligible.
        if result.get("zone_risk_eligible", True):
            save_report_to_db(result, output_name, incident_time)
        
    except Exception as exc:
        return render_template(
            "upload.html",
            error=f"Processing failed: {exc}",
        )

    return render_template("result.html", result=result, video_filename=output_name)

@app.route("/processed_videos/<path:filename>")
def processed_video(filename: str):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

# History route for viewing archived road safety reports
@app.route("/history")
def history():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    reports = cursor.execute('SELECT * FROM safety_reports ORDER BY timestamp DESC').fetchall()
    conn.close()
    return render_template("history.html", reports=reports)

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Fetch all records to map out the city
    reports = cursor.execute('SELECT * FROM safety_reports').fetchall()
    conn.close()
    
    # Convert rows to a list of dicts so it's easier to handle in JavaScript
    report_list = [dict(row) for row in reports]
    return render_template("dashboard.html", reports=report_list)
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return render_template(
        "upload.html",
        error="File too large. Max upload size is 500 MB.",
    ), 413

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
