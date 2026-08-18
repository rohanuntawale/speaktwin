#!/usr/bin/env python
"""Download the public posture datasets used by the posture-model pipeline.

No Kaggle account is required for these public datasets.  The script uses
Kaggle's public dataset API, records the source metadata, and extracts ZIP
archives with a path-traversal guard.

    python scripts/download_posture_data.py
    python scripts/download_posture_data.py --dataset cctv
    python scripts/download_posture_data.py --data-dir data/posture
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {
    "cctv": {
        "ref": "cctvdataset/cctv-exam-monitor-dataset",
        "license": "CC0: Public Domain",
        "description": "Real classroom/exam images with correct, forward, backward, left and right movement labels.",
    },
    "keypoints": {
        "ref": "melsmm/posture-keypoints-detection",
        "license": "Apache 2.0",
        "description": "Sitting/standing posture images with YOLO pose keypoint labels.",
    },
    "silhouettes": {
        "ref": "mexwell/silhouettes-for-human-posture-recognition",
        "license": "CC BY 4.0",
        "description": "Sitting, standing, bending and lying silhouette images.",
    },
    "wlu": {
        "ref": "sulaimanmuhammed/wlu-rehabilitation-posture",
        "license": "MIT",
        "description": "Real rehabilitation exercise videos organised by correct/incorrect form.",
    },
}


def api_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "SpeakTwin posture data downloader"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "SpeakTwin posture data downloader"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if total:
                print(f"\r  {copied / total:6.1%}  {copied / 1024**2:8.1f}/{total / 1024**2:.1f} MB", end="", flush=True)
        print()


def safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            destination = (target / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        zipped.extractall(target)


def fetch(name: str, data_root: Path) -> dict:
    spec = DATASETS[name]
    ref = spec["ref"]
    metadata = api_json(f"https://www.kaggle.com/api/v1/datasets/view/{ref}")
    version = metadata.get("currentVersionNumber", 1)
    slug = ref.split("/", 1)[1]
    folder = data_root / name
    archive = data_root / "archives" / f"{name}-v{version}.zip"
    extracted = folder / f".extracted-v{version}"

    print(f"[{name}] {metadata.get('title', slug)}")
    print(f"  license: {metadata.get('licenseName') or spec['license']}")
    print(f"  size: {metadata.get('totalBytes', 0) / 1024**2:.1f} MB")
    print(f"  source: https://www.kaggle.com/datasets/{ref}")

    if not archive.exists():
        print("  downloading...")
        download(
            f"https://www.kaggle.com/api/v1/datasets/download/{ref}?datasetVersionNumber={version}",
            archive,
        )
    else:
        print("  archive already exists; reusing it")

    if not extracted.exists():
        print("  extracting...")
        safe_extract(archive, folder)
        extracted.touch()
    else:
        print("  extraction already exists; reusing it")

    return {
        "name": name,
        "ref": ref,
        "version": version,
        "title": metadata.get("title"),
        "license": metadata.get("licenseName") or spec["license"],
        "description": spec["description"],
        "source_url": f"https://www.kaggle.com/datasets/{ref}",
        "archive": str(archive),
        "folder": str(folder),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/posture"))
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    try:
        for name in names:
            manifest.append(fetch(name, args.data_dir))
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        return 1

    manifest_path = args.data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nComplete. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
