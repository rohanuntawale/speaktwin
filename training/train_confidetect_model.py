#!/usr/bin/env python
"""Train the auxiliary ConfiDetect posture/confidence classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("data/posture/confidetect/confidence_features_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/posture_online"))
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    target = "confidence_label"
    features = [column for column in df.columns if column != target]
    categorical = [column for column in features if df[column].dtype == "object"]
    numeric = [column for column in features if column not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    train, test = train_test_split(
        range(len(df)), test_size=0.2, stratify=df[target], random_state=42,
    )
    pipeline.fit(df.iloc[train][features], df.iloc[train][target])
    predicted = pipeline.predict(df.iloc[test][features])
    print({
        "accuracy": accuracy_score(df.iloc[test][target], predicted),
        "balanced_accuracy": balanced_accuracy_score(df.iloc[test][target], predicted),
    })
    print(classification_report(df.iloc[test][target], predicted))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "features": features, "target": target,
                 "source": "ConfiDetect MIT"},
                args.output_dir / "confidetect_classifier.joblib", compress=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
