#!/usr/bin/env python
"""
SpeakTwin - Confidence Weight Fitting
======================================
Replaces the guessed weights in `CONFIDENCE_WEIGHTS` with values fitted
against human delivery ratings.

The problem being solved
------------------------
`backend/services/confidence_score.py` combines four sub-scores:

    score = 100 * (0.25*wpm + 0.25*pitch_var + 0.20*energy + 0.30*fluency)

Those four numbers are a plausible prior. Nobody measured them. This script
keeps the sub-score *functions* - they encode real domain knowledge about
what good delivery sounds like - and fits only the **weights**, against
speechocean762's human ratings.

Fitting weights rather than end-to-end regression keeps the result
interpretable and droppable straight back into the config.

Usage
-----
    python scripts/datasets.py download speechocean762     # 409 MB, no account
    python training/train_confidence_scorer.py \
        --data-dir data/speechocean762/speechocean762 \
        --target fluency

    # Faster iteration on a subset:
    python training/train_confidence_scorer.py --limit 500

Honest limitation
-----------------
speechocean762 is *read* speech: short prompted utterances from L2 English
speakers, scored for pronunciation. That means:

  * filler rate is ~0 in nearly every clip, so the fluency weight is fitted
    on almost no signal and should be treated as unreliable
  * utterances are a few seconds long, so WPM is noisy
  * the raters were judging pronunciation proficiency, not public speaking

The fitted weights are therefore a better-grounded starting point than the
guess, not a finished answer. Weights fitted on spontaneous, rated public
speaking would be materially better - no such corpus is openly available.
The script prints this caveat with its results so it cannot be forgotten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from backend.services.audio_analysis import analyze_audio  # noqa: E402
from backend.services.audio_io import prepare  # noqa: E402
from backend.services.confidence_score import (  # noqa: E402
    _energy_score,
    _filler_score,
    _pitch_variation_score,
    _wpm_score,
)
from backend.services.filler_detection import detect_fillers  # noqa: E402
from backend.utils.helpers import SAMPLE_RATE  # noqa: E402

FEATURES = ["wpm", "pitch_variation", "energy", "filler_usage"]
# speechocean762 scores utterances 0-10 on these axes.
TARGETS = ("fluency", "prosodic", "accuracy", "completeness")
SCORE_MAX = 10.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_index(data_dir: Path) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    """Read scores.json plus the transcript and wav path tables."""
    scores_path = data_dir / "resource" / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"No scores.json at {scores_path}.\n"
            f"Run: python scripts/datasets.py download speechocean762"
        )

    scores = json.loads(scores_path.read_text(encoding="utf-8"))

    text: Dict[str, str] = {}
    wavs: Dict[str, str] = {}
    for split in ("train", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        text_file = split_dir / "text"
        if text_file.exists():
            for line in text_file.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    utt, transcript = line.split("\t", 1)
                    text[utt.strip()] = transcript.strip()
        scp = split_dir / "wav.scp"
        if scp.exists():
            for line in scp.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    utt, path = line.split("\t", 1)
                    wavs[utt.strip()] = path.strip()

    return scores, text, wavs


def extract_features(audio: np.ndarray, transcript: str) -> Optional[Dict[str, float]]:
    """
    Run SpeakTwin's own pipeline over one utterance.

    Deliberately reuses the production analysis code: weights fitted against
    different features than the ones the server computes would not transfer.
    """
    duration = len(audio) / float(SAMPLE_RATE)
    if duration < 0.4:
        return None

    metrics = analyze_audio(audio, SAMPLE_RATE)
    fillers = detect_fillers(transcript)
    words = fillers["total_words"]
    if words == 0:
        return None

    wpm = words / (duration / 60.0)

    return {
        "wpm": _wpm_score(wpm),
        "pitch_variation": _pitch_variation_score(metrics["pitch_std"]),
        "energy": _energy_score(metrics["energy_db"]),
        "filler_usage": _filler_score(float(fillers["filler_rate"])),
        # Kept for the diagnostics table, not used as model inputs.
        "_raw_wpm": wpm,
        "_raw_pitch_std": metrics["pitch_std"],
        "_raw_energy_db": metrics["energy_db"],
        "_raw_filler_rate": float(fillers["filler_rate"]),
    }


def build_dataset(data_dir: Path, target: str,
                  limit: Optional[int]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    scores, text, wavs = load_index(data_dir)

    rows: List[Dict[str, Any]] = []
    skipped = {"no_audio": 0, "no_score": 0, "no_features": 0, "decode": 0}

    utterances = list(scores.keys())
    print(f"Found {len(utterances)} scored utterances")

    for index, utt in enumerate(utterances):
        if limit and len(rows) >= limit:
            break
        if index % 500 == 0 and index:
            print(f"  processed {index}, usable {len(rows)}")

        entry = scores[utt]
        if target not in entry:
            skipped["no_score"] += 1
            continue

        relative = wavs.get(utt)
        if not relative:
            skipped["no_audio"] += 1
            continue

        wav_path = data_dir / relative
        if not wav_path.exists():
            skipped["no_audio"] += 1
            continue

        try:
            audio, _ = prepare(wav_path.read_bytes(), max_seconds=30.0)
        except Exception:
            skipped["decode"] += 1
            continue

        features = extract_features(audio, text.get(utt, ""))
        if features is None:
            skipped["no_features"] += 1
            continue

        features["_target"] = float(entry[target]) / SCORE_MAX
        features["_utt"] = utt
        rows.append(features)

    print(f"\nUsable: {len(rows)}   Skipped: {skipped}")
    if len(rows) < 30:
        raise RuntimeError(
            f"Only {len(rows)} usable utterances - too few to fit anything. "
            f"Check the dataset extracted correctly."
        )

    X = np.array([[row[f] for f in FEATURES] for row in rows], dtype=np.float64)
    y = np.array([row["_target"] for row in rows], dtype=np.float64)
    return X, y, rows


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_weights(X: np.ndarray, y: np.ndarray, seed: int = 42
                ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit non-negative weights that sum to 1.

    Constrained deliberately: a negative weight would mean "speaking at the
    optimal rate makes you sound worse", which is not a claim this data can
    support and not something to ship into a coaching product.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = LinearRegression(positive=True, fit_intercept=True)
    model.fit(X_train, y_train)

    raw = np.maximum(model.coef_, 0.0)
    total = raw.sum()
    if total <= 0:
        raise RuntimeError(
            "Every fitted coefficient is zero - the sub-scores carry no signal "
            "for this target. Try --target prosodic."
        )
    weights = {name: float(w / total) for name, w in zip(FEATURES, raw)}

    predicted = model.predict(X_test)
    metrics = {
        "r2": float(model.score(X_test, y_test)),
        "mae": float(np.mean(np.abs(predicted - y_test))),
        "baseline_mae": float(np.mean(np.abs(y_test - y_train.mean()))),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return weights, metrics


def current_weights() -> Dict[str, float]:
    from backend.utils.helpers import CONFIDENCE_WEIGHTS
    return {
        "wpm": CONFIDENCE_WEIGHTS["wpm"],
        "pitch_variation": CONFIDENCE_WEIGHTS["pitch_variation"],
        "energy": CONFIDENCE_WEIGHTS["energy"],
        "filler_usage": CONFIDENCE_WEIGHTS["filler_penalty"],
    }


def report(weights: Dict[str, float], metrics: Dict[str, float],
           rows: List[Dict], target: str) -> None:
    current = current_weights()

    print("\n" + "=" * 66)
    print(f"FITTED WEIGHTS   (target: {target})")
    print("=" * 66)
    print(f"{'Feature':<20} {'Current':>10} {'Fitted':>10} {'Change':>10}")
    print("-" * 66)
    for name in FEATURES:
        delta = weights[name] - current[name]
        print(f"{name:<20} {current[name]:>10.3f} {weights[name]:>10.3f} "
              f"{delta:>+10.3f}")

    print("\nFit quality")
    print("-" * 66)
    print(f"  R²             {metrics['r2']:.3f}"
          f"   (1.0 = perfect, 0.0 = no better than the mean)")
    print(f"  MAE            {metrics['mae']:.3f}   vs baseline "
          f"{metrics['baseline_mae']:.3f}")
    print(f"  Train / test   {metrics['n_train']} / {metrics['n_test']}")

    # Feature sanity: a near-constant feature cannot support its weight,
    # however confident the regression looks. This check is the difference
    # between a fitted number and a meaningful one.
    print("\nFeature spread (a flat feature makes its weight meaningless)")
    print("-" * 66)
    unreliable: List[str] = []
    for name in FEATURES:
        values = np.array([r[name] for r in rows])
        flat = values.std() < 0.05
        if flat:
            unreliable.append(name)
        flag = "  <-- nearly constant, weight NOT trustworthy" if flat else ""
        print(f"  {name:<18} mean {values.mean():.3f}  std {values.std():.3f}{flag}")

    filler_rates = np.array([r["_raw_filler_rate"] for r in rows])
    zero_share = 100 * (filler_rates == 0).mean()
    print(f"\n  raw filler rate: mean {filler_rates.mean():.4f}, "
          f"{zero_share:.0f}% of clips have zero fillers")
    if zero_share > 80 and "filler_usage" not in unreliable:
        unreliable.append("filler_usage")

    print("\n" + "=" * 66)
    if metrics["r2"] < 0.1:
        print("VERDICT: weak fit. These sub-scores explain little of the human")
        print("rating on this corpus. Do NOT adopt these weights.")
        return

    if unreliable:
        print("VERDICT: PARTIAL. The overall fit is usable, but these features")
        print("carry too little variation in this corpus for their weights to")
        print("mean anything:")
        for name in unreliable:
            print(f"    - {name}  (fitted {weights[name]:.3f}, keep "
                  f"{current[name]:.3f})")
        print("\nAdopt only the trustworthy weights, renormalised so the four")
        print("still sum to 1.0:")
        blended = {n: (current[n] if n in unreliable else weights[n])
                   for n in FEATURES}
        total = sum(blended.values())
        blended = {n: v / total for n, v in blended.items()}
    else:
        print("VERDICT: usable. Every feature carries variation.")
        blended = weights

    print("\n    CONFIDENCE_WEIGHTS = {")
    print(f'        "wpm": {blended["wpm"]:.2f},')
    print(f'        "pitch_variation": {blended["pitch_variation"]:.2f},')
    print(f'        "energy": {blended["energy"]:.2f},')
    print(f'        "filler_penalty": {blended["filler_usage"]:.2f},')
    print("    }")

    print("\nCAVEAT: speechocean762 is read speech scored for pronunciation,")
    print("not spontaneous public speaking. Filler rate is ~0 in almost every")
    print("clip, so its weight is fitted on nearly no signal. Treat this as a")
    print("better-grounded prior than the guess, not a finished answer.")
    print("=" * 66)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit SpeakTwin confidence weights against human ratings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", default="data/speechocean762/speechocean762")
    parser.add_argument("--target", default="fluency", choices=TARGETS,
                        help="which human rating to fit against")
    parser.add_argument("--limit", type=int,
                        help="cap utterances processed, for quick iteration")
    parser.add_argument("--save", help="write the fitted weights to a JSON file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Not found: {data_dir}")
        print("Run: python scripts/datasets.py download speechocean762")
        return 1

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn is required: pip install -r requirements-ml.txt")
        return 1

    X, y, rows = build_dataset(data_dir, args.target, args.limit)
    weights, metrics = fit_weights(X, y)
    report(weights, metrics, rows, args.target)

    if args.save:
        Path(args.save).write_text(
            json.dumps({"target": args.target, "weights": weights,
                        "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
