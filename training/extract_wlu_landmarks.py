#!/usr/bin/env python
"""Convert labelled WLU rehabilitation videos to posture feature CSV.

Only the WLU ``Sit To Stand`` subset is used. Correct videos become
``correct`` rows; incorrect videos become ``uncertain`` rows because the
dataset does not say that every incorrect rehabilitation movement is a
specific speaking-posture error.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backend.services.pose_analysis import PoseFrame, analyse_frame


FEATURES = [
    "shoulder_tilt", "head_tilt", "torso_lean", "openness", "head_scale",
    "head_offset", "forward_head", "hand_face_dist", "head_drop",
]
LANDMARK_NOSE = 0
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12


def sample_video(path: Path, label: str, max_frames: int) -> list[dict]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
    aspect = width / height
    indices = set(round(i * max(total - 1, 0) / max_frames) for i in range(max_frames))
    rows = []
    raw = []
    options = python.BaseOptions(model_asset_path=str(Path("data/posture/wlu/pose_landmarker_full.task").resolve()))
    detector_options = vision.PoseLandmarkerOptions(
        base_options=options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    with vision.PoseLandmarker.create_from_options(detector_options) as detector:
        for index in range(total):
            ok, frame = capture.read()
            if not ok:
                break
            if index not in indices:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.pose_landmarks:
                continue
            points = result.pose_landmarks[0]
            landmarks = [
                {"x": p.x, "y": p.y, "z": p.z, "visibility": getattr(p, "visibility", 1.0)}
                for p in points
            ]
            metrics = analyse_frame(PoseFrame(landmarks, aspect))
            if metrics is None:
                continue
            left = landmarks[LANDMARK_LEFT_SHOULDER]
            right = landmarks[LANDMARK_RIGHT_SHOULDER]
            nose = landmarks[LANDMARK_NOSE]
            shoulder_width = ((left["x"] - right["x"]) ** 2 + (left["y"] - right["y"]) ** 2) ** 0.5
            head_ratio = (((left["y"] + right["y"]) / 2) - nose["y"]) / max(shoulder_width, 1e-6)
            raw.append((metrics, max(head_ratio, 1e-5), index / fps))
    capture.release()
    if not raw:
        return []
    baseline = max(item[1] for item in raw)
    for metrics, head_ratio, timestamp in raw:
        rows.append({
            "label": label,
            "subject": f"wlu-{path.stem}",
            "split": "train",
            **{key: metrics.get(key, 0.8 if key == "hand_face_dist" else 0.0) for key in FEATURES[:-1]},
            "head_drop": head_ratio / baseline,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/posture/wlu/Blurred"))
    parser.add_argument("--output", type=Path, default=Path("data/posture/real_landmarks.csv"))
    parser.add_argument("--frames-per-video", type=int, default=12)
    args = parser.parse_args()

    rows = []
    for folder in ("Sit To Stand Correct", "Sit To Stand Incorrect"):
        label = "correct" if folder.endswith("Correct") else "uncertain"
        for video in sorted((args.root / folder).glob("*.mp4")):
            rows.extend(sample_video(video, label, args.frames_per_video))
            print(f"processed {video.name}: {len(rows)} rows total")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "subject", "split", *FEATURES])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
