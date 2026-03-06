import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from detection import process_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed_videos"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["PROCESSED_FOLDER"] = str(PROCESSED_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    if request.method == "GET":
        return render_template("upload.html")

    video = request.files.get("video")
    latitude = request.form.get("latitude", type=float)
    longitude = request.form.get("longitude", type=float)
    location_name = request.form.get("location_name", type=str, default="")

    if not video or video.filename == "":
        return render_template("upload.html", error="Please choose a video file.")

    if not allowed_file(video.filename):
        return render_template("upload.html", error="Unsupported file type. Use MP4/MOV/AVI/MKV.")

    if latitude is None or longitude is None:
        return render_template("upload.html", error="Please pick a location on the map.")

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
        result = process_video(
            input_video_path=str(upload_path),
            output_video_path=str(output_path),
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
        )
    except Exception as exc:
        return render_template(
            "upload.html",
            error=f"Processing failed: {exc}",
        )

    return render_template("result.html", result=result, video_filename=output_name)


@app.route("/processed_videos/<path:filename>")
def processed_video(filename: str):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return render_template(
        "upload.html",
        error="File too large. Max upload size is 500 MB.",
    ), 413


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
