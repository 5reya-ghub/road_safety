from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# COCO classes used:
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASS_IDS = {0, 1, 2, 3, 5, 7}
CLASS_NAME_MAP = {
    0: "Pedestrian",
    1: "Bike",
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck",
}


@dataclass
class Track:
    track_id: int
    class_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    last_frame: int
    prev_centroid: Optional[Tuple[float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)  # pixels/second


class SimpleCentroidTracker:
    def __init__(self, max_distance: float = 60.0, max_missed_frames: int = 20) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Dict], frame_idx: int, fps: float) -> Dict[int, Track]:
        if not detections:
            self._remove_stale(frame_idx)
            return self.tracks

        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))
        candidate_matches: List[Tuple[int, int, float]] = []

        for track_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                dist = euclidean(track.centroid, det["centroid"])
                if dist <= self.max_distance:
                    candidate_matches.append((track_id, det_idx, dist))

        candidate_matches.sort(key=lambda x: x[2])
        accepted_matches: List[Tuple[int, int]] = []
        for track_id, det_idx, _ in candidate_matches:
            if track_id in unmatched_tracks and det_idx in unmatched_detections:
                accepted_matches.append((track_id, det_idx))
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(det_idx)

        for track_id, det_idx in accepted_matches:
            det = detections[det_idx]
            track = self.tracks[track_id]

            prev_centroid = track.centroid
            dt = max((frame_idx - track.last_frame) / max(fps, 1e-6), 1e-6)
            vx = (det["centroid"][0] - prev_centroid[0]) / dt
            vy = (det["centroid"][1] - prev_centroid[1]) / dt

            track.prev_centroid = prev_centroid
            track.centroid = det["centroid"]
            track.bbox = det["bbox"]
            track.class_id = det["class_id"]
            track.last_frame = frame_idx
            track.velocity = (vx, vy)

        for det_idx in unmatched_detections:
            det = detections[det_idx]
            self.tracks[self.next_track_id] = Track(
                track_id=self.next_track_id,
                class_id=det["class_id"],
                centroid=det["centroid"],
                bbox=det["bbox"],
                last_frame=frame_idx,
            )
            self.next_track_id += 1

        self._remove_stale(frame_idx)
        return self.tracks

    def _remove_stale(self, frame_idx: int) -> None:
        stale_ids = [
            tid
            for tid, track in self.tracks.items()
            if (frame_idx - track.last_frame) > self.max_missed_frames
        ]
        for tid in stale_ids:
            del self.tracks[tid]


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def calculate_ttc(
    pos_a: Tuple[float, float],
    vel_a: Tuple[float, float],
    pos_b: Tuple[float, float],
    vel_b: Tuple[float, float],
) -> Tuple[float, float, float]:
    distance = euclidean(pos_a, pos_b)
    if distance < 1e-6:
        return distance, 0.0, 0.0

    rel_vx = vel_a[0] - vel_b[0]
    rel_vy = vel_a[1] - vel_b[1]
    rel_px = pos_a[0] - pos_b[0]
    rel_py = pos_a[1] - pos_b[1]

    # Closing speed along line-of-sight. If <= 0, objects are not approaching.
    closing_speed = -((rel_px * rel_vx) + (rel_py * rel_vy)) / distance
    if closing_speed <= 1e-6:
        return distance, 0.0, float("inf")

    ttc = distance / closing_speed
    return distance, closing_speed, ttc


def is_valid_pair(class_a: int, class_b: int) -> bool:
    # Ignore pedestrian-pedestrian proximity; keep all other pairs.
    return not (class_a == 0 and class_b == 0)


def is_vulnerable_pair(class_a: int, class_b: int) -> bool:
    vulnerable = {0, 1, 3}
    return (class_a in vulnerable) or (class_b in vulnerable)


def bbox_edge_distance(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    dx = max(ax1 - bx2, bx1 - ax2, 0)
    dy = max(ay1 - by2, by1 - ay2, 0)
    return math.hypot(dx, dy)


def classify_risk(near_miss_count: int, duration_s: float, events: List[Dict]) -> Tuple[str, Dict]:
    safe_duration = max(duration_s, 1.0)
    events_per_min = (near_miss_count * 60.0) / safe_duration
    severe_events = sum(1 for e in events if e.get("ttc_s", 99.0) <= 0.6)
    risk_score = (0.8 * near_miss_count) + (0.25 * events_per_min) + (1.2 * severe_events)

    if near_miss_count >= 6 or events_per_min >= 15 or severe_events >= 4:
        return "HIGH", {
            "events_per_min": round(events_per_min, 2),
            "severe_events": severe_events,
            "risk_score": round(risk_score, 2),
        }
    if near_miss_count >= 3 or events_per_min >= 6 or severe_events >= 2:
        return "MEDIUM", {
            "events_per_min": round(events_per_min, 2),
            "severe_events": severe_events,
            "risk_score": round(risk_score, 2),
        }
    return "LOW", {
        "events_per_min": round(events_per_min, 2),
        "severe_events": severe_events,
        "risk_score": round(risk_score, 2),
    }


def risk_color(risk_level: str) -> str:
    return {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(risk_level, "blue")


def detect_objects(model: YOLO, frame: np.ndarray) -> List[Dict]:
    results = model.predict(
        source=frame,
        verbose=False,
        conf=0.35,
        imgsz=736,
        classes=sorted(TARGET_CLASS_IDS),
    )
    detections: List[Dict] = []
    for result in results:
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for bbox, class_id in zip(xyxy, cls_ids):
            if class_id not in TARGET_CLASS_IDS:
                continue
            x1, y1, x2, y2 = bbox.astype(int)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append(
                {"class_id": class_id, "bbox": (x1, y1, x2, y2), "centroid": (cx, cy)}
            )
    return detections


def process_video(
    input_video_path: str,
    output_video_path: str,
    latitude: float,
    longitude: float,
    location_name: str = "",
    yolo_model_path: str = "yolov8n.pt",
    distance_threshold: float = 60.0,
    edge_distance_threshold: float = 28.0,
    ttc_threshold: float = 1.4,
    min_relative_speed: float = 12.0,
    event_cooldown_frames: int = 20,
    process_every_n_frames: int = 1,
    max_inference_width: int = 1280,
    min_risky_streak_frames: int = 2,
    highlight_persist_frames: int = 22,
) -> Dict:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-3 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out_path = out_path.with_name(f"{out_path.stem}_raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = YOLO(yolo_model_path)
    tracker = SimpleCentroidTracker(max_distance=70.0, max_missed_frames=25)

    frame_idx = 0
    near_miss_count = 0
    near_miss_events: List[Dict] = []
    pair_last_triggered: Dict[Tuple[int, int], int] = {}
    pair_risky_streak: Dict[Tuple[int, int], int] = {}
    track_highlight_until: Dict[int, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if max_inference_width > 0 and frame.shape[1] > max_inference_width:
            scale = max_inference_width / frame.shape[1]
            infer_frame = cv2.resize(
                frame,
                (max_inference_width, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale = 1.0
            infer_frame = frame

        should_infer = (frame_idx % max(process_every_n_frames, 1) == 0)
        detections: List[Dict] = []
        if should_infer:
            raw_detections = detect_objects(model, infer_frame)
            if scale != 1.0:
                inv = 1.0 / scale
                for d in raw_detections:
                    x1, y1, x2, y2 = d["bbox"]
                    cx, cy = d["centroid"]
                    detections.append(
                        {
                            "class_id": d["class_id"],
                            "bbox": (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)),
                            "centroid": (cx * inv, cy * inv),
                        }
                    )
            else:
                detections = raw_detections

        tracks = tracker.update(detections, frame_idx, fps)

        active_ids = [tid for tid, tr in tracks.items() if (frame_idx - tr.last_frame) <= 1]
        active_tracks = [(tid, tracks[tid]) for tid in active_ids]
        near_miss_ids = set()

        risky_candidates: List[Tuple[float, float, int, int, Track, Track, float]] = []
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                id_a, tr_a = active_tracks[i]
                id_b, tr_b = active_tracks[j]
                if not is_valid_pair(tr_a.class_id, tr_b.class_id):
                    continue

                distance, relative_speed, ttc = calculate_ttc(
                    tr_a.centroid,
                    tr_a.velocity,
                    tr_b.centroid,
                    tr_b.velocity,
                )
                edge_distance = bbox_edge_distance(tr_a.bbox, tr_b.bbox)
                # Ignore very low-motion pairs and pedestrian-pedestrian proximity.
                if relative_speed < min_relative_speed:
                    continue
                if tr_a.class_id == 0 and tr_b.class_id == 0:
                    continue

                pair_is_vulnerable = is_vulnerable_pair(tr_a.class_id, tr_b.class_id)
                pair_distance_threshold = distance_threshold + (10.0 if pair_is_vulnerable else 0.0)
                pair_edge_threshold = edge_distance_threshold + (8.0 if pair_is_vulnerable else 0.0)
                pair_ttc_threshold = ttc_threshold + (0.2 if pair_is_vulnerable else 0.0)

                is_close = (distance < pair_distance_threshold) or (edge_distance < pair_edge_threshold)
                is_risky = is_close and ttc < pair_ttc_threshold

                pair_key = tuple(sorted((id_a, id_b)))
                if is_risky:
                    pair_risky_streak[pair_key] = pair_risky_streak.get(pair_key, 0) + 1
                else:
                    pair_risky_streak[pair_key] = 0

                if is_risky and pair_risky_streak.get(pair_key, 0) >= min_risky_streak_frames:
                    risky_candidates.append(
                        (ttc, edge_distance, id_a, id_b, tr_a, tr_b, relative_speed, distance)
                    )

        risky_candidates.sort(key=lambda x: (x[0], x[1]))
        for ttc, edge_distance, id_a, id_b, tr_a, tr_b, relative_speed, distance in risky_candidates:
            near_miss_ids.update([id_a, id_b])
            track_highlight_until[id_a] = max(track_highlight_until.get(id_a, -1), frame_idx + highlight_persist_frames)
            track_highlight_until[id_b] = max(track_highlight_until.get(id_b, -1), frame_idx + highlight_persist_frames)

            pair_key = tuple(sorted((id_a, id_b)))
            last_trigger = pair_last_triggered.get(pair_key, -10_000)
            if (frame_idx - last_trigger) > event_cooldown_frames:
                near_miss_count += 1
                pair_last_triggered[pair_key] = frame_idx
                near_miss_events.append(
                    {
                        "frame": frame_idx,
                        "object_a": CLASS_NAME_MAP.get(tr_a.class_id, str(tr_a.class_id)),
                        "id_a": id_a,
                        "object_b": CLASS_NAME_MAP.get(tr_b.class_id, str(tr_b.class_id)),
                        "id_b": id_b,
                        "distance_px": round(distance, 2),
                        "edge_distance_px": round(edge_distance, 2),
                        "relative_speed_px_s": round(relative_speed, 2),
                        "ttc_s": round(ttc, 2),
                        "timestamp_s": round(frame_idx / max(fps, 1e-6), 2),
                    }
                )

            p1 = (int(tr_a.centroid[0]), int(tr_a.centroid[1]))
            p2 = (int(tr_b.centroid[0]), int(tr_b.centroid[1]))
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"NEAR MISS TTC:{ttc:.2f}s",
                (min(p1[0], p2[0]), max(min(p1[1], p2[1]) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        for track_id, track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            class_name = CLASS_NAME_MAP.get(track.class_id, str(track.class_id))
            persist_red = frame_idx <= track_highlight_until.get(track_id, -1)
            color = (0, 0, 255) if (track_id in near_miss_ids or persist_red) else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{class_name} ID:{track_id}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cv2.putText(
            frame,
            f"Near Miss Events: {near_miss_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    _ensure_browser_playable(raw_out_path, out_path)

    duration_s = frame_idx / max(fps, 1e-6)
    risk_level, risk_metrics = classify_risk(near_miss_count, duration_s, near_miss_events)
    return {
        "location_name": location_name.strip() or "User-selected location",
        "latitude": latitude,
        "longitude": longitude,
        "near_miss_count": near_miss_count,
        "risk_level": risk_level,
        "risk_marker_color": risk_color(risk_level),
        "risk_score": risk_metrics["risk_score"],
        "events_per_min": risk_metrics["events_per_min"],
        "severe_events": risk_metrics["severe_events"],
        "distance_threshold_px": distance_threshold,
        "edge_distance_threshold_px": edge_distance_threshold,
        "ttc_threshold_s": ttc_threshold,
        "total_frames": frame_idx,
        "duration_s": round(duration_s, 2),
        "events": near_miss_events,
    }


def _ensure_browser_playable(raw_path: Path, final_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_path),
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=180)
        if completed.returncode == 0 and final_path.exists() and final_path.stat().st_size > 0:
            raw_path.unlink(missing_ok=True)
            return
    except Exception:
        pass

    if final_path.exists():
        final_path.unlink()
    raw_path.replace(final_path)
