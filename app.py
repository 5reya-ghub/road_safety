import uuid
import sqlite3
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, redirect, url_for
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
            rash_event_count INTEGER DEFAULT 0,
            video_risk_score REAL DEFAULT 0,
            risk_level TEXT,
            risk_score REAL,
            video_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Backward-compatible schema updates for existing databases.
    for alter in (
        "ALTER TABLE safety_reports ADD COLUMN pedestrian_conflict_count INTEGER DEFAULT 0",
        "ALTER TABLE safety_reports ADD COLUMN rash_event_count INTEGER DEFAULT 0",
        "ALTER TABLE safety_reports ADD COLUMN video_risk_score REAL DEFAULT 0",
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
    conn.commit()
    conn.close()


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
    rash_event_count = int(result.get("rash_event_count", 0) or 0)

    # Event severity scoring (user policy):
    # near miss = 2, pedestrian conflict = 3
    # Additional signal: rash-driving single-vehicle events = 2 (kept moderate).
    video_risk_score = (2 * non_ped_near_miss) + (3 * pedestrian_conflicts) + (2 * rash_event_count)
    return {
        "pedestrian_conflict_count": pedestrian_conflicts,
        "rash_event_count": rash_event_count,
        "video_risk_score": float(video_risk_score),
    }


# Save processed results to the persistent database
def save_report_to_db(result, video_filename):
    metrics = compute_video_risk_metrics(result)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO safety_reports 
        (location_name, latitude, longitude, near_miss_count, pedestrian_conflict_count, rash_event_count, video_risk_score, risk_level, risk_score, video_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result.get('location_name', 'Unknown'),
        result.get('latitude'),
        result.get('longitude'),
        result.get('near_miss_count'),
        metrics["pedestrian_conflict_count"],
        metrics["rash_event_count"],
        metrics["video_risk_score"],
        result.get('risk_level'),
        result.get('risk_score'), # Supported by detection.py
        video_filename
    ))
    conn.commit()
    conn.close()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_location_status(number_of_videos: int, location_risk_score):
    if number_of_videos < 3:
        return "Insufficient Data", "#94A3B8"
    if location_risk_score is None:
        return "Insufficient Data", "#94A3B8"
    if 0.0 <= location_risk_score < 1.0:
        return "Safe", "#26D07C"
    if 1.0 <= location_risk_score < 3.0:
        return "Moderate Risk", "#00D1FF"
    if 3.0 <= location_risk_score < 5.0:
        return "High Risk", "#FFCC00"
    return "Dangerous", "#FF2E63"

@app.route("/")
def index():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Group videos by nearby coordinates rounded to 3 decimals (~110m).
    query = '''
        SELECT 
            ROUND(latitude, 3) as zone_lat, 
            ROUND(longitude, 3) as zone_lng,
            MAX(location_name) as location_name,
            COUNT(*) as number_of_videos,
            SUM(near_miss_count) as total_misses,
            SUM(pedestrian_conflict_count) as total_pedestrian_conflicts,
            SUM(rash_event_count) as total_rash_events,
            SUM(video_risk_score) as total_video_risk_score
        FROM safety_reports
        GROUP BY zone_lat, zone_lng
        ORDER BY number_of_videos DESC
    '''
    zones = cursor.execute(query).fetchall()
    conn.close()

    # Location-level scoring policy:
    # If number_of_videos < 3 => Insufficient Data
    # Else location_risk_score = sum(video_risk_score) / number_of_videos
    # Buckets: 0-1 Safe, 1-3 Moderate Risk, 3-5 High Risk, 5+ Dangerous.
    zone_list = []
    for zone in zones:
        z = dict(zone)
        number_of_videos = int(z.get("number_of_videos", 0) or 0)
        total_video_risk_score = float(z.get("total_video_risk_score", 0.0) or 0.0)
        location_risk_score = None
        if number_of_videos >= 3:
            location_risk_score = total_video_risk_score / number_of_videos

        status, color = classify_location_status(number_of_videos, location_risk_score)
        z["zone_risk"] = status
        z["color"] = color
        z["location_risk_score"] = round(location_risk_score, 2) if location_risk_score is not None else None
        zone_list.append(z)

    return render_template("index.html", zones=zone_list)

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
            save_report_to_db(result, output_name)
        
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
    init_db() # Ensure DB and table exist before starting
    app.run(debug=True, host="0.0.0.0", port=5000)
