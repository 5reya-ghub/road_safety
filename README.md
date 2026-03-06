# AI Near-Miss Road Safety Detection System

Hackathon-ready Flask prototype for near-miss road safety detection using uploaded videos, map-tagged location, YOLOv8 detection, centroid tracking, and TTC-based risk detection.

## Features
- Upload traffic video (MP4/MOV/AVI/MKV)
- Tag video location on Leaflet map (lat/lng)
- YOLOv8 detection for car, bike, bus, truck, pedestrian
- Simple ID-based object tracking
- TTC near-miss detection: `TTC = distance / relative_speed`
- Annotated output video with alerts
- Risk scoring: LOW / MEDIUM / HIGH
- Results dashboard with map marker color by risk

## Project Structure
```text
/project
├── app.py
├── detection.py
├── templates
│   ├── index.html
│   ├── upload.html
│   ├── result.html
├── static
│   ├── css
│   │   └── styles.css
│   ├── js
│   │   ├── upload.js
│   │   └── result.js
├── uploads
├── processed_videos
└── requirements.txt
```

## Run Locally
1. Create and activate a virtual environment.
   - Windows PowerShell:
   - `python -m venv .venv`
   - `.\\.venv\\Scripts\\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start the app:
   - `python app.py`
4. Open:
   - `http://127.0.0.1:5000`

## Notes
- On first run, `ultralytics` downloads `yolov8n.pt`.
- Near-miss parameters can be tuned in `process_video()` inside `detection.py`:
  - `distance_threshold` in pixels
  - `ttc_threshold` in seconds
- This prototype uses pixel-space distance/speed and is intended for hackathon demo use.
