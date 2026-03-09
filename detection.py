from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# COCO classes used for Road Safety
TARGET_CLASS_IDS = {0, 1, 2, 3, 5, 7}
MOTOR_VEHICLE_CLASS_IDS = {2, 3, 5, 7}
CLASS_NAME_MAP = {
    0: "Pedestrian",
    1: "Bike",
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck",
}
_MODEL_CACHE: Dict[str, YOLO] = {}

@dataclass
class Track:
    track_id: int
    class_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    last_frame: int
    age_frames: int = 1
    prev_centroid: Optional[Tuple[float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)

class SimpleCentroidTracker:
    def __init__(
        self,
        max_distance: float = 60.0,
        max_missed_frames: int = 20,
        velocity_smoothing: float = 0.35,
        max_velocity_px_s: float = 220.0,
        min_motion_px: float = 1.2,
    ) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.velocity_smoothing = min(max(velocity_smoothing, 0.0), 1.0)
        self.max_velocity_px_s = max(max_velocity_px_s, 1.0)
        self.min_motion_px = max(min_motion_px, 0.0)
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
            dx = det["centroid"][0] - prev_centroid[0]
            dy = det["centroid"][1] - prev_centroid[1]
            
            raw_vx = dx / dt if math.hypot(dx, dy) >= self.min_motion_px else 0.0
            raw_vy = dy / dt if math.hypot(dx, dy) >= self.min_motion_px else 0.0

            alpha = self.velocity_smoothing
            vx = (alpha * raw_vx) + ((1.0 - alpha) * track.velocity[0])
            vy = (alpha * raw_vy) + ((1.0 - alpha) * track.velocity[1])
            
            speed = math.hypot(vx, vy)
            if speed > self.max_velocity_px_s:
                scale = self.max_velocity_px_s / max(speed, 1e-6)
                vx *= scale; vy *= scale

            track.prev_centroid = prev_centroid
            track.centroid = det["centroid"]
            track.bbox = det["bbox"]
            track.class_id = det["class_id"]
            track.last_frame = frame_idx
            track.age_frames += 1
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
        stale_ids = [tid for tid, t in self.tracks.items() if (frame_idx - t.last_frame) > self.max_missed_frames]
        for tid in stale_ids: del self.tracks[tid]

# --- Helper Functions ---
def euclidean(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def calculate_ttc(pos_a, vel_a, pos_b, vel_b):
    dist = euclidean(pos_a, pos_b)
    if dist < 1e-6: return dist, 0.0, 0.0
    rel_v = (vel_a[0] - vel_b[0], vel_a[1] - vel_b[1])
    rel_p = (pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
    closing_speed = -((rel_p[0] * rel_v[0]) + (rel_p[1] * rel_v[1])) / dist
    if closing_speed <= 1e-6: return dist, 0.0, float("inf")
    return dist, closing_speed, dist / closing_speed

def is_valid_pair(ca, cb): return not (ca == 0 and cb == 0)
def is_vulnerable_pair(ca, cb): return ca in {0, 1, 3} or cb in {0, 1, 3}

def bbox_edge_distance(b1, b2):
    dx = max(b1[0] - b2[2], b2[0] - b1[2], 0)
    dy = max(b1[1] - b2[3], b2[1] - b1[3], 0)
    return math.hypot(dx, dy)


def angle_delta_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def compute_rash_event_score(
    speed: float,
    accel: float,
    heading_change_deg: float,
    speed_threshold: float,
    accel_threshold: float,
    turn_threshold_deg: float,
) -> float:
    speed_factor = min(2.0, speed / max(speed_threshold, 1e-6))
    accel_factor = min(2.0, accel / max(accel_threshold, 1e-6))
    turn_factor = min(2.0, heading_change_deg / max(turn_threshold_deg, 1e-6))
    score = (45.0 * speed_factor) + (35.0 * accel_factor) + (20.0 * turn_factor)
    return float(min(100.0, max(0.0, score)))


def aggregate_rash_score(rash_events: List[Dict]) -> float:
    if not rash_events:
        return 0.0
    avg_event_score = sum(e.get("rash_score", 0.0) for e in rash_events) / len(rash_events)
    density_boost = min(1.0, len(rash_events) / 4.0)
    return round(min(100.0, avg_event_score * (0.75 + 0.25 * density_boost)), 2)

def classify_risk(count, dur, events):
    epm = (count * 60.0) / max(dur, 1.0)
    severe = sum(1 for e in events if e.get("ttc_s", 99.0) <= 0.6)
    score = (0.8 * count) + (0.25 * epm) + (1.2 * severe)
    # Calibrated to avoid one-level-under labeling in short urban clips.
    if count >= 20 or epm >= 28 or severe >= 8 or score >= 30:
        return "HIGH", {"risk_score": round(score, 2), "events_per_min": round(epm, 2), "severe_events": severe}
    if count >= 6 or epm >= 10 or severe >= 3 or score >= 12:
        return "MEDIUM", {"risk_score": round(score, 2), "events_per_min": round(epm, 2), "severe_events": severe}
    return "LOW", {"risk_score": round(score, 2), "events_per_min": round(epm, 2), "severe_events": severe}

def risk_color(level): return {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(level, "blue")

def detect_objects(model, frame):
    results = model.predict(source=frame, verbose=False, conf=0.35, imgsz=640, classes=sorted(TARGET_CLASS_IDS))
    dets = []
    for r in results:
        if r.boxes is None: continue
        for box, cid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box.astype(int)
            dets.append({"class_id": cid, "bbox": (x1, y1, x2, y2), "centroid": ((x1+x2)/2.0, (y1+y2)/2.0)})
    return dets


def get_yolo_model(model_path: str) -> YOLO:
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached
    model = YOLO(model_path)
    _MODEL_CACHE[model_path] = model
    return model

def process_video(
    input_video_path, output_video_path, latitude, longitude, location_name="",
    yolo_model_path="yolov8n.pt", distance_threshold=60.0, edge_distance_threshold=28.0,
    ttc_threshold=0.85, min_relative_speed=14.0, event_cooldown_frames=22, process_every_n_frames=3,
    max_inference_width=800, min_risky_streak_frames=3, pair_rearm_safe_frames=6, safe_ttc_rearm_margin=0.8,
    min_track_age_frames=6, min_bbox_area_px=900.0, max_relative_speed_px_s=220.0,
    min_distance_reduction_px=2.5, motion_pixel_ratio_threshold=0.002, motion_min_area_px=500.0,
    motion_bg_history=400, motion_bg_var_threshold=16.0, motion_warmup_frames=20,
    segment_merge_gap_frames=10, highlight_persist_frames=22,
    rash_speed_threshold=70.0, rash_accel_threshold=140.0, rash_turn_threshold_deg=35.0,
    rash_min_streak_frames=4, rash_event_cooldown_frames=40, rash_min_track_age_frames=8
):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = Path(output_video_path)
    raw_out_path = out_path.with_name(f"{out_path.stem}_raw.mp4")
    writer = cv2.VideoWriter(str(raw_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    model = get_yolo_model(yolo_model_path)
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=motion_bg_history, varThreshold=motion_bg_var_threshold, detectShadows=False)
    tracker = SimpleCentroidTracker(max_distance=85.0, max_missed_frames=25, max_velocity_px_s=max_relative_speed_px_s)

    frame_idx, near_miss_count = 0, 0
    near_miss_events, pair_last_triggered, pair_risky_streak, pair_safe_streak, pair_event_armed = [], {}, {}, {}, {}
    track_highlight_until = {}
    track_prev_speed, track_prev_heading = {}, {}
    track_rash_streak, track_last_rash_frame, track_rash_highlight_until = {}, {}, {}
    rash_events = []
    frames_motion, frames_skipped, frames_infer = 0, 0, 0
    first_motion_frame = None

    while True:
        ok, frame = cap.read()
        if not ok: break

        should_infer = (frame_idx % process_every_n_frames == 0)
        scale = 1.0
        infer_frame = None
        if should_infer:
            scale = max_inference_width / frame.shape[1] if frame.shape[1] > max_inference_width else 1.0
            infer_frame = cv2.resize(frame, (max_inference_width, int(frame.shape[0]*scale))) if scale != 1.0 else frame

        motion_confirmed = True
        if should_infer:
            mask = bg_sub.apply(infer_frame)
            if frame_idx >= motion_warmup_frames:
                _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
                moving_ratio = cv2.countNonZero(mask) / mask.size
                motion_confirmed = moving_ratio >= motion_pixel_ratio_threshold
                if motion_confirmed:
                    frames_motion += 1
                    if first_motion_frame is None:
                        first_motion_frame = frame_idx
                else:
                    frames_skipped += 1

        detections = []
        if should_infer and motion_confirmed:
            raw_dets = detect_objects(model, infer_frame)
            frames_infer += 1
            inv = 1.0 / scale
            for d in raw_dets:
                x1, y1, x2, y2 = d["bbox"]
                cx, cy = d["centroid"]
                detections.append({"class_id": d["class_id"], "bbox": (int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv)), "centroid": (cx*inv, cy*inv)})

        tracks = tracker.update(detections, frame_idx, fps)
        # Keep tracks visible between sparse inference steps.
        max_track_visual_gap = max(1, process_every_n_frames + 1)
        active_ids = [tid for tid, tr in tracks.items() if (frame_idx - tr.last_frame) <= max_track_visual_gap]
        active_tracks = [(tid, tracks[tid]) for tid in active_ids]
        near_miss_ids = set()
        rash_ids = set()

        # Single-vehicle rash driving detection.
        for tid, tr in active_tracks:
            if tr.class_id not in MOTOR_VEHICLE_CLASS_IDS or tr.age_frames < rash_min_track_age_frames:
                continue

            speed = math.hypot(tr.velocity[0], tr.velocity[1])
            prev_speed = track_prev_speed.get(tid, speed)
            accel = abs(speed - prev_speed) * max(fps, 1e-6)

            heading = track_prev_heading.get(tid, 0.0)
            if speed > 1e-6:
                heading = math.degrees(math.atan2(tr.velocity[1], tr.velocity[0]))
            prev_heading = track_prev_heading.get(tid, heading)
            heading_change = angle_delta_deg(heading, prev_heading) if speed > 10.0 and prev_speed > 10.0 else 0.0

            rash_condition = speed >= rash_speed_threshold and (
                accel >= rash_accel_threshold or heading_change >= rash_turn_threshold_deg
            )
            if rash_condition:
                track_rash_streak[tid] = track_rash_streak.get(tid, 0) + 1
            else:
                track_rash_streak[tid] = 0

            last_rash_frame = track_last_rash_frame.get(tid, -10_000)
            if (
                rash_condition
                and track_rash_streak.get(tid, 0) >= rash_min_streak_frames
                and (frame_idx - last_rash_frame) > rash_event_cooldown_frames
            ):
                rash_score = compute_rash_event_score(
                    speed=speed,
                    accel=accel,
                    heading_change_deg=heading_change,
                    speed_threshold=rash_speed_threshold,
                    accel_threshold=rash_accel_threshold,
                    turn_threshold_deg=rash_turn_threshold_deg,
                )
                rash_events.append(
                    {
                        "frame": frame_idx,
                        "id": tid,
                        "object": CLASS_NAME_MAP.get(tr.class_id, str(tr.class_id)),
                        "timestamp_s": round(frame_idx / max(fps, 1e-6), 2),
                        "speed_px_s": round(speed, 2),
                        "accel_px_s2": round(accel, 2),
                        "heading_change_deg": round(heading_change, 2),
                        "rash_score": round(rash_score, 2),
                    }
                )
                track_last_rash_frame[tid] = frame_idx
                track_rash_highlight_until[tid] = frame_idx + highlight_persist_frames

            if frame_idx <= track_rash_highlight_until.get(tid, -1):
                rash_ids.add(tid)

            track_prev_speed[tid] = speed
            track_prev_heading[tid] = heading

        # Collision Calculation Logic
        risky_candidates = []
        for i in range(len(active_tracks)):
            for j in range(i+1, len(active_tracks)):
                id_a, tr_a = active_tracks[i]; id_b, tr_b = active_tracks[j]
                if not is_valid_pair(tr_a.class_id, tr_b.class_id) or tr_a.age_frames < min_track_age_frames or tr_b.age_frames < min_track_age_frames: continue
                
                dist, rel_speed, ttc = calculate_ttc(tr_a.centroid, tr_a.velocity, tr_b.centroid, tr_b.velocity)
                edge_dist = bbox_edge_distance(tr_a.bbox, tr_b.bbox)
                
                pair_vulnerable = is_vulnerable_pair(tr_a.class_id, tr_b.class_id)
                p_dist_t = distance_threshold + 10 if pair_vulnerable else distance_threshold
                p_ttc_t = ttc_threshold + 0.05 if pair_vulnerable else ttc_threshold
                
                is_risky = (dist < p_dist_t or edge_dist < edge_distance_threshold) and ttc < p_ttc_t
                
                if is_risky and tr_a.prev_centroid and tr_b.prev_centroid:
                    if (euclidean(tr_a.prev_centroid, tr_b.prev_centroid) - dist) < min_distance_reduction_px: is_risky = False

                pair_key = tuple(sorted((id_a, id_b)))
                if is_risky: pair_risky_streak[pair_key] = pair_risky_streak.get(pair_key, 0) + 1
                else: pair_risky_streak[pair_key] = 0
                
                if not is_risky and dist > p_dist_t*1.1: pair_safe_streak[pair_key] = pair_safe_streak.get(pair_key, 0) + 1
                else: pair_safe_streak[pair_key] = 0
                if pair_safe_streak.get(pair_key, 0) >= pair_rearm_safe_frames: pair_event_armed[pair_key] = True

                if is_risky and pair_event_armed.get(pair_key, True) and pair_risky_streak.get(pair_key, 0) >= min_risky_streak_frames:
                    risky_candidates.append((ttc, edge_dist, id_a, id_b, tr_a, tr_b, rel_speed, dist))

        for ttc, edge_d, id_a, id_b, tr_a, tr_b, rel_s, d in sorted(risky_candidates):
            near_miss_ids.update([id_a, id_b])
            pair_key = tuple(sorted((id_a, id_b)))
            pair_event_armed[pair_key], pair_safe_streak[pair_key] = False, 0
            track_highlight_until[id_a] = frame_idx + highlight_persist_frames
            track_highlight_until[id_b] = frame_idx + highlight_persist_frames
            if (frame_idx - pair_last_triggered.get(pair_key, -10000)) > event_cooldown_frames:
                near_miss_count += 1
                pair_last_triggered[pair_key] = frame_idx
                near_miss_events.append({"frame": frame_idx, "object_a": CLASS_NAME_MAP[tr_a.class_id], "id_a": id_a, "object_b": CLASS_NAME_MAP[tr_b.class_id], "id_b": id_b, "ttc_s": round(ttc, 2), "timestamp_s": round(frame_idx/fps, 2)})
            cv2.line(frame, (int(tr_a.centroid[0]), int(tr_a.centroid[1])), (int(tr_b.centroid[0]), int(tr_b.centroid[1])), (0, 0, 255), 2)

        for tid, t in active_tracks:
            color = (255, 0, 255) if tid in rash_ids else (0, 255, 0)
            if tid in near_miss_ids or frame_idx <= track_highlight_until.get(tid, -1):
                color = (0, 0, 255)
            cv2.rectangle(frame, (t.bbox[0], t.bbox[1]), (t.bbox[2], t.bbox[3]), color, 2)
            if tid in rash_ids and tid not in near_miss_ids:
                cv2.putText(
                    frame,
                    "RASH",
                    (t.bbox[0], max(t.bbox[1] - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                )

        writer.write(frame)
        frame_idx += 1

    dur = frame_idx / fps
    risk_level, metrics = classify_risk(near_miss_count, dur, near_miss_events)
    rash_risk_score = aggregate_rash_score(rash_events)
    rash_driving_detected = len(rash_events) > 0
    zone_risk_eligible = not (near_miss_count == 0 and rash_driving_detected)
    cap.release(); writer.release(); _ensure_browser_playable(raw_out_path, out_path)
    
    return {
        "location_name": location_name, "latitude": latitude, "longitude": longitude,
        "near_miss_count": near_miss_count, "risk_level": risk_level, 
        "risk_marker_color": risk_color(risk_level), # Added for map consistency
        "risk_score": metrics["risk_score"], "events_per_min": metrics["events_per_min"],
        "severe_events": metrics["severe_events"], "duration_s": round(dur, 2), "events": near_miss_events,
        "rash_driving_detected": rash_driving_detected,
        "rash_event_count": len(rash_events),
        "rash_risk_score": rash_risk_score,
        "rash_events": rash_events,
        "zone_risk_eligible": zone_risk_eligible,
        "zone_risk_note": (
            "Single-vehicle rash driving detected. Location excluded from zone risk map."
            if not zone_risk_eligible
            else ""
        ),
        "motion_start_s": (
            round(first_motion_frame / max(fps, 1e-6), 2)
            if first_motion_frame is not None
            else None
        ),
    }

def _ensure_browser_playable(raw, final):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw),
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
            str(final),
        ],
        capture_output=True,
    )
    if final.exists(): raw.unlink()
