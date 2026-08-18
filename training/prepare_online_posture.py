#!/usr/bin/env python
"""Convert the downloaded MultiPosture MediaPipe CSV to SpeakTwin features.

The source labels are mapped conservatively:
TUP -> correct, TLF -> sunk, TLL/TLR -> leaning, TLB -> uncertain.
The source contains no explicit forward-head label, so it is never fabricated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.pose_analysis import PoseFrame, analyse_frame


FEATURES = [
    "shoulder_tilt", "head_tilt", "torso_lean", "openness", "head_scale",
    "head_offset", "forward_head", "hand_face_dist", "head_drop",
]
LABEL_MAP = {"TUP": "correct", "TLF": "sunk", "TLL": "leaning",
             "TLR": "leaning", "TLB": "uncertain"}

LANDMARK_NAMES = {
    0: "nose", 7: "left_ear", 8: "right_ear", 11: "left_shoulder",
    12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee", 27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel", 31: "left_foot_index",
    32: "right_foot_index",
}


def landmarks(row: dict[str, str]) -> list[dict[str, float]]:
    result = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for _ in range(33)]
    for index, name in LANDMARK_NAMES.items():
        result[index] = {
            "x": float(row[f"{name}_x"]),
            "y": float(row[f"{name}_y"]),
            "z": float(row[f"{name}_z"]),
            "visibility": 1.0,
        }
    return result


def convert(source: Path, output: Path) -> int:
    grouped: dict[str, list[tuple[dict[str, str], dict]]] = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = LABEL_MAP.get(row.get("upperbody_label", ""))
            if not label:
                continue
            metrics = analyse_frame(PoseFrame(landmarks(row), 1.0))
            if metrics is not None:
                grouped[row["subject"]].append((row, metrics))

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "subject", "split", *FEATURES]
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject, items in sorted(grouped.items()):
            ratios = []
            for row, _ in items:
                left = landmarks(row)[11]
                right = landmarks(row)[12]
                nose = landmarks(row)[0]
                width = max(((left["x"] - right["x"]) ** 2 +
                             (left["y"] - right["y"]) ** 2) ** 0.5, 1e-6)
                ratios.append(((left["y"] + right["y"]) / 2 - nose["y"]) / width)
            baseline = max(ratios) if ratios else 1.0
            for (row, metrics), ratio in zip(items, ratios):
                writer.writerow({
                    "label": LABEL_MAP[row["upperbody_label"]],
                    "subject": f"multiposture-{subject}",
                    "split": "train",
                    **{key: metrics.get(key, 0.8 if key == "hand_face_dist" else 0.0)
                       for key in FEATURES[:-1]},
                    "head_drop": ratio / max(baseline, 1e-6),
                })
                count += 1
    print(f"wrote {count} rows to {output}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("data/posture/multiposture/data.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/posture/multiposture_landmarks.csv"))
    args = parser.parse_args()
    raise SystemExit(0 if convert(args.source, args.output) else 1)
