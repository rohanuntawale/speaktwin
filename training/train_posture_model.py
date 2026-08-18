#!/usr/bin/env python
"""Train a conservative posture-error classifier on MediaPipe-style features.

The live app judges landmark-derived features, so this trainer uses the same
kind of feature vector instead of trying to learn from unrelated image
pixels. Synthetic samples are generated with controlled perturbations. Real
samples may be supplied as CSV with these columns:

    label,subject,split,shoulder_tilt,head_tilt,torso_lean,openness,
    head_scale,head_offset,forward_head,hand_face_dist,head_drop

`split` must be `train` or `test`; when omitted, subjects are split by group.

Examples:
    python training/train_posture_model.py --synthetic-only
    python training/train_posture_model.py --real-csv data/posture/real_landmarks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit


FEATURES = [
    "shoulder_tilt",
    "head_tilt",
    "torso_lean",
    "openness",
    "head_scale",
    "head_offset",
    "forward_head",
    "hand_face_dist",
    "head_drop",
]
LABELS = ["correct", "sunk", "forward_head", "leaning", "uncertain"]
MIN_REAL_PER_LABEL = 20
MIN_REAL_SUBJECTS = 3


def normal_sample(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    return rng.normal(mean, sd, size).astype(np.float32)


def synthetic_data(n_per_label: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create feature-level perturbations that match webcam measurement noise."""
    rng = np.random.default_rng(seed)
    rows: List[np.ndarray] = []
    labels: List[str] = []

    for label in LABELS:
        n = n_per_label
        shoulder_tilt = normal_sample(rng, 3.0, 2.0, n)
        head_tilt = normal_sample(rng, 3.0, 2.5, n)
        torso_lean = normal_sample(rng, 5.0, 2.5, n)
        openness = normal_sample(rng, 0.78, 0.05, n)
        head_scale = normal_sample(rng, 0.42, 0.025, n)
        head_offset = normal_sample(rng, 0.0, 0.04, n)
        forward_head = normal_sample(rng, 0.04, 0.04, n)
        hand_face_dist = normal_sample(rng, 0.8, 0.25, n)
        head_drop = normal_sample(rng, 1.0, 0.025, n)

        if label == "sunk":
            torso_lean = normal_sample(rng, 13.0, 3.0, n)
            openness = normal_sample(rng, 0.54, 0.07, n)
            head_drop = normal_sample(rng, 0.72, 0.06, n)
        elif label == "forward_head":
            head_scale = normal_sample(rng, 0.52, 0.035, n)
            forward_head = normal_sample(rng, 0.20, 0.06, n)
        elif label == "leaning":
            shoulder_tilt = normal_sample(rng, 14.0, 3.0, n)
            torso_lean = normal_sample(rng, 12.0, 3.0, n)
            head_offset = normal_sample(rng, 0.17, 0.07, n)
        elif label == "uncertain":
            # Occlusion, low visibility, and borderline poses must not trigger
            # a correction. This is deliberately broad and conservative.
            shoulder_tilt = normal_sample(rng, 8.0, 7.0, n)
            head_tilt = normal_sample(rng, 8.0, 7.0, n)
            torso_lean = normal_sample(rng, 9.0, 7.0, n)
            openness = normal_sample(rng, 0.66, 0.16, n)
            head_scale = normal_sample(rng, 0.46, 0.10, n)
            head_drop = normal_sample(rng, 0.90, 0.15, n)

        features = np.column_stack([
            shoulder_tilt, head_tilt, torso_lean, openness, head_scale,
            head_offset, forward_head, hand_face_dist, head_drop,
        ])
        rows.append(features)
        labels.extend([label] * n)

    X = np.vstack(rows).astype(np.float32)
    y = np.asarray(labels)
    groups = np.arange(len(y)) // 5
    order = rng.permutation(len(y))
    return X[order], y[order], groups[order]


def read_real_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Real landmark CSV not found: {path}. "
            "Create it from labelled webcam sessions using "
            "data/posture/real_landmarks.template.csv, or run with "
            "--synthetic-only for the prototype."
        )
    rows: List[List[float]] = []
    labels: List[str] = []
    groups: List[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [key for key in ["label", *FEATURES] if key not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            label = row["label"].strip()
            if label not in LABELS:
                raise ValueError(f"Unknown label {label!r}; expected one of {LABELS}")
            rows.append([float(row[key]) for key in FEATURES])
            labels.append(label)
            groups.append(row.get("subject") or row.get("session") or str(len(groups)))
    if not rows:
        raise ValueError(f"{path} contains no labelled rows")
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    too_small = [f"{label}={count}" for label, count in counts.items()
                 if count < MIN_REAL_PER_LABEL]
    if too_small:
        raise ValueError(
            f"{path} has too few real rows ({', '.join(too_small)}). "
            f"Collect at least {MIN_REAL_PER_LABEL} labelled rows per class "
            f"from at least {MIN_REAL_SUBJECTS} people before the 50/50 run."
        )
    subject_count = len(set(groups))
    if subject_count < MIN_REAL_SUBJECTS:
        raise ValueError(
            f"{path} has only {subject_count} subject(s). Collect labelled "
            f"sessions from at least {MIN_REAL_SUBJECTS} people."
        )
    return np.asarray(rows, dtype=np.float32), np.asarray(labels), np.asarray(groups)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(
        # Tuned on grouped holdouts. More trees stabilise borderline webcam
        # frames; a smaller leaf preserves the narrow forward-head boundary.
        n_estimators=1000,
        min_samples_leaf=1,
        max_features=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-csv", type=Path)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--synthetic-per-label", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, default=Path("models/posture"))
    args = parser.parse_args()

    if args.real_csv and args.synthetic_only:
        parser.error("choose --real-csv or --synthetic-only, not both")
    if not args.real_csv and not args.synthetic_only:
        parser.error("provide --real-csv for the 50/50 run, or --synthetic-only for a prototype")

    X_syn, y_syn, groups_syn = synthetic_data(args.synthetic_per_label, seed=42)
    if args.real_csv:
        try:
            X_real, y_real, groups_real = read_real_csv(args.real_csv)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        rng = np.random.default_rng(42)
        n = min(len(X_syn), len(X_real))
        syn_idx = rng.choice(len(X_syn), n, replace=False)
        real_idx = rng.choice(len(X_real), n, replace=False)
        X = np.vstack([X_syn[syn_idx], X_real[real_idx]])
        y = np.concatenate([y_syn[syn_idx], y_real[real_idx]])
        groups = np.concatenate([[f"synthetic-{g}" for g in groups_syn[syn_idx]], groups_real[real_idx]])
        source = "50% synthetic / 50% real"
    else:
        X, y, groups = X_syn, y_syn, groups_syn
        source = "synthetic prototype only"

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    model = train_model(X[train_idx], y[train_idx])
    predicted = model.predict(X[test_idx])
    report = classification_report(y[test_idx], predicted, output_dict=True, zero_division=0)
    metrics = {
        "source": source,
        "n_samples": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "labels": LABELS,
        "features": FEATURES,
        "classification_report": report,
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], predicted)),
        "confusion_matrix": confusion_matrix(y[test_idx], predicted, labels=LABELS).tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Evaluation above is held out by subject/video. Refit the deliverable on
    # every eligible row only after measuring it, so production benefits from
    # all downloaded data without contaminating the reported benchmark.
    model = train_model(X, y)
    # Compression keeps the production artifact practical without changing
    # predictions or inference behavior.
    joblib.dump({"model": model, "features": FEATURES, "labels": LABELS},
                args.output_dir / "posture_classifier.joblib", compress=3)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.output_dir / 'posture_classifier.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
