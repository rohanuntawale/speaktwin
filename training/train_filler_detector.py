#!/usr/bin/env python
"""
SpeakTwin - Acoustic Filler Detector Training
==============================================
Fine-tunes a self-supervised speech backbone (WavLM / wav2vec2) into a
frame-level classifier that finds filler words in the waveform.

Why this model has to exist
---------------------------
Whisper was trained largely on cleaned transcripts and routinely deletes
"um" and "uh"; enabling its VAD filter trims the hesitation regions too.
So counting fillers in the transcript structurally under-counts, and no
regex improvement fixes it - the evidence only survives in the audio.

Usage
-----
    # 1. Get the data (needs the PodcastFillers request form first)
    python scripts/datasets.py info podcastfillers

    # 2. Train
    python training/train_filler_detector.py \
        --data-dir data/podcastfillers \
        --output-dir models/filler-detector \
        --epochs 3

    # 3. Point the backend at it
    #    ML_DISFLUENCY_ENABLED=true
    #    ML_DISFLUENCY_MODEL=./models/filler-detector

On a single modern GPU this is hours, not days. On CPU it is not
practical - use Kaggle (~30 free GPU-hours/week), Colab, or an hourly
rental.

Dataset layout
--------------
The loader expects a manifest CSV with one row per audio clip:

    audio_path,label,start,end
    clips/ep001_0001.wav,um,1.24,1.51
    clips/ep001_0002.wav,none,0.00,2.50

`--manifest` overrides the default of `<data-dir>/manifest.csv`. Adapt
`build_manifest_from_podcastfillers()` if the release layout differs from
what you downloaded - that function is the only dataset-specific part.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Index 0 must stay the negative class so argmax==0 means "nothing here".
LABELS = ["none", "um", "uh", "repetition", "prolongation"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

SAMPLE_RATE = 16_000
# 20 ms frames match the stride of a wav2vec2/WavLM encoder.
FRAME_SECONDS = 0.02


@dataclass
class Example:
    audio_path: Path
    label: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def read_manifest(path: Path, data_dir: Path) -> List[Example]:
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}.\n"
            f"Create one, or run with --build-manifest to derive it from a "
            f"PodcastFillers checkout."
        )

    examples: List[Example] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].strip().lower()
            if label not in LABEL_TO_ID:
                label = "none"
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = data_dir / audio_path
            examples.append(Example(
                audio_path=audio_path,
                label=label,
                start=float(row.get("start", 0.0) or 0.0),
                end=float(row.get("end", 0.0) or 0.0),
            ))
    return examples


def build_manifest_from_podcastfillers(data_dir: Path, output: Path) -> int:
    """
    Derive a manifest from a PodcastFillers checkout.

    The release ships per-clip audio plus event CSVs. Exact column names
    have varied between versions, so this is written defensively and
    reports what it found rather than assuming. Adjust the column names
    below to match the release you actually downloaded.
    """
    clips_dir = next((p for p in (data_dir / "audio", data_dir / "clips",
                                  data_dir) if p.is_dir()), None)
    if clips_dir is None:
        raise FileNotFoundError(f"No audio directory under {data_dir}")

    label_files = list(data_dir.rglob("*.csv"))
    if not label_files:
        raise FileNotFoundError(
            f"No annotation CSV found under {data_dir}. "
            f"Check the download completed and extracted."
        )

    rows: List[Dict[str, Any]] = []
    for label_file in label_files:
        if label_file.resolve() == output.resolve():
            continue
        with label_file.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                lowered = {k.lower().strip(): v for k, v in row.items() if k}
                label = (lowered.get("label") or lowered.get("event_type")
                         or lowered.get("class") or "none")
                clip = (lowered.get("clip_name") or lowered.get("filename")
                        or lowered.get("audio_path"))
                if not clip:
                    continue
                rows.append({
                    "audio_path": str(Path(clips_dir.name) / clip),
                    "label": str(label).strip().lower(),
                    "start": lowered.get("start_time") or lowered.get("start") or 0.0,
                    "end": lowered.get("end_time") or lowered.get("end") or 0.0,
                })

    if not rows:
        raise RuntimeError(
            "Parsed the annotation files but produced no rows - the column "
            "names differ from what this script expects. Inspect one CSV and "
            "update build_manifest_from_podcastfillers()."
        )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "label", "start", "end"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_hf_dataset(examples: List[Example], max_seconds: float):
    """Turn the manifest into a HF dataset of waveforms + frame labels."""
    import numpy as np
    import soundfile as sf
    from datasets import Dataset

    max_samples = int(max_seconds * SAMPLE_RATE)
    frames_per_clip = int(max_seconds / FRAME_SECONDS)

    def generator():
        for example in examples:
            if not example.audio_path.exists():
                continue
            try:
                audio, sr = sf.read(example.audio_path, dtype="float32",
                                    always_2d=False)
            except Exception:
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                from scipy.signal import resample_poly
                import math
                divisor = math.gcd(sr, SAMPLE_RATE)
                audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor)

            audio = np.asarray(audio, dtype=np.float32)[:max_samples]
            if audio.size < max_samples:
                audio = np.pad(audio, (0, max_samples - audio.size))

            # Frame labels: positive only inside the annotated span.
            labels = np.zeros(frames_per_clip, dtype=np.int64)
            if example.label != "none" and example.end > example.start:
                first = max(0, int(example.start / FRAME_SECONDS))
                last = min(frames_per_clip, int(example.end / FRAME_SECONDS) + 1)
                labels[first:last] = LABEL_TO_ID[example.label]

            yield {"input_values": audio, "labels": labels}

    return Dataset.from_generator(generator)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args) -> int:
    import numpy as np
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModelForAudioFrameClassification,
        Trainer,
        TrainingArguments,
    )

    data_dir = Path(args.data_dir).resolve()
    manifest = Path(args.manifest) if args.manifest else data_dir / "manifest.csv"

    if args.build_manifest:
        count = build_manifest_from_podcastfillers(data_dir, manifest)
        print(f"Wrote {count} rows to {manifest}")

    examples = read_manifest(manifest, data_dir)
    print(f"Loaded {len(examples)} examples")

    distribution: Dict[str, int] = {}
    for example in examples:
        distribution[example.label] = distribution.get(example.label, 0) + 1
    print("Label distribution:", distribution)

    if len(distribution) < 2:
        print("ERROR: manifest contains only one class - nothing to learn.")
        return 1

    dataset = build_hf_dataset(examples, args.max_seconds)
    splits = dataset.train_test_split(test_size=args.eval_fraction, seed=42)

    config = AutoConfig.from_pretrained(
        args.backbone,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label={i: l for l, i in LABEL_TO_ID.items()},
    )
    model = AutoModelForAudioFrameClassification.from_pretrained(
        args.backbone, config=config, ignore_mismatched_sizes=True,
    )
    extractor = AutoFeatureExtractor.from_pretrained(args.backbone)

    if args.freeze_encoder:
        # Cheap first pass: train only the head. Good for a quick signal
        # check before committing GPU hours to a full fine-tune.
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()

    def compute_metrics(prediction):
        logits, labels = prediction
        predicted = np.argmax(logits, axis=-1).reshape(-1)
        truth = labels.reshape(-1)

        accuracy = float((predicted == truth).mean())
        # Frame labels are overwhelmingly "none", so accuracy alone is
        # misleading; report positive-class F1 as well.
        positive_pred = predicted != 0
        positive_true = truth != 0
        true_positive = int((positive_pred & positive_true).sum())
        precision = true_positive / max(1, int(positive_pred.sum()))
        recall = true_positive / max(1, int(positive_true.sum()))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        return {"accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=args.fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nFinal metrics:", metrics)

    trainer.save_model(args.output_dir)
    extractor.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")
    print("Enable it with:")
    print("  ML_DISFLUENCY_ENABLED=true")
    print(f"  ML_DISFLUENCY_MODEL={args.output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune an acoustic filler-word detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", help="defaults to <data-dir>/manifest.csv")
    parser.add_argument("--build-manifest", action="store_true",
                        help="derive the manifest from a PodcastFillers checkout")
    parser.add_argument("--output-dir", default="models/filler-detector")
    parser.add_argument("--backbone", default="microsoft/wavlm-base-plus",
                        help="wav2vec2/WavLM/HuBERT checkpoint to fine-tune")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-seconds", type=float, default=2.5,
                        help="clip length; match the backend's chunk size")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="GPU only")
    return train(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
