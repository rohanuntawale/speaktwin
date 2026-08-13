#!/usr/bin/env python
"""
SpeakTwin - Dataset Manager
============================
Inspect, verify, and download the speech corpora SpeakTwin trains on.

    python scripts/datasets.py list
    python scripts/datasets.py list --task filler_detection
    python scripts/datasets.py info podcastfillers
    python scripts/datasets.py check
    python scripts/datasets.py download speechocean762
    python scripts/datasets.py download librispeech --subset dev-clean

What this tool will NOT do, by design: create accounts, or accept dataset
licences on your behalf. Several corpora here (Common Voice, TED-LIUM,
AMI, PodcastFillers, IEMOCAP) require clicking "I agree" to a licence
under your own name. That is a legal act, so it stays yours. `check` tells
you exactly which ones are waiting on you.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dataset_registry import (  # noqa: E402
    DATASETS,
    Access,
    Dataset,
    Source,
    by_task,
)

DATA_ROOT = Path(os.getenv("SPEAKTWIN_DATA_DIR", "data")).resolve()

ACCESS_LABEL = {
    Access.OPEN: "open        ",
    Access.HF_AUTH: "HF login    ",
    Access.REQUEST: "request form",
    Access.PAID: "paid (LDC)  ",
}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def human_gb(value: float) -> str:
    return f"{value:.1f} GB" if value >= 1 else f"{int(value * 1024)} MB"


def free_space_gb(path: Path) -> float:
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    return shutil.disk_usage(target).free / (1024 ** 3)


def hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_list(args) -> int:
    entries = by_task(args.task) if args.task else list(DATASETS.values())
    if args.open_only:
        entries = [d for d in entries if d.access == Access.OPEN]

    if not entries:
        print(f"No datasets match task '{args.task}'.")
        return 1

    print(f"\n{'KEY':<18} {'ACCESS':<13} {'SIZE':>8}  {'TASK':<28} NAME")
    print("-" * 100)
    for dataset in sorted(entries, key=lambda d: (not d.recommended, d.key)):
        star = "*" if dataset.recommended else " "
        print(f"{star}{dataset.key:<17} {ACCESS_LABEL[dataset.access]} "
              f"{human_gb(dataset.size_gb):>8}  {dataset.task:<28} {dataset.name}")

    print("\n* = recommended starting point for SpeakTwin")
    print(f"Data directory: {DATA_ROOT}")
    print(f"Free space:     {free_space_gb(DATA_ROOT):.1f} GB\n")
    return 0


def cmd_info(args) -> int:
    dataset = DATASETS.get(args.key)
    if dataset is None:
        print(f"Unknown dataset '{args.key}'. Run `list` to see the options.")
        return 1

    print(f"\n{dataset.name}  [{dataset.key}]")
    print("=" * 70)
    print(f"Task        : {dataset.task}")
    print(f"Access      : {dataset.access.value}")
    print(f"Source      : {dataset.source.value}")
    print(f"Size        : ~{human_gb(dataset.size_gb)}")
    print(f"Licence     : {dataset.licence}")
    print(f"Homepage    : {dataset.url}")
    if dataset.hf_id:
        config = f" (config: {dataset.hf_config})" if dataset.hf_config else ""
        print(f"HF dataset  : {dataset.hf_id}{config}")
    print(f"\n{dataset.description}")
    if dataset.notes:
        print(f"\nNote: {dataset.notes}")

    print("\nHow to get it:")
    print(_instructions(dataset))
    print()
    return 0


def _instructions(dataset: Dataset) -> str:
    if dataset.access == Access.OPEN:
        return f"  python scripts/datasets.py download {dataset.key}"

    if dataset.access == Access.HF_AUTH:
        return (
            f"  1. Create a free account at https://huggingface.co/join\n"
            f"  2. Open https://huggingface.co/datasets/{dataset.hf_id} and\n"
            f"     accept the terms (this is a licence agreement - it has to\n"
            f"     be you, it cannot be automated)\n"
            f"  3. Create a token at https://huggingface.co/settings/tokens\n"
            f"  4. Put HF_TOKEN=<token> in your .env\n"
            f"  5. python scripts/datasets.py download {dataset.key}"
        )

    if dataset.access == Access.REQUEST:
        return (
            f"  1. Request access at {dataset.url}\n"
            f"  2. Wait for approval (hours to days)\n"
            f"  3. Download their archive into {DATA_ROOT / dataset.key}"
        )

    return (
        f"  1. Obtain an LDC membership or purchase the corpus:\n"
        f"     {dataset.url}\n"
        f"  2. Extract into {DATA_ROOT / dataset.key}"
    )


def cmd_check(args) -> int:
    """Report what is present, what is blocked, and on what."""
    print(f"\nData directory : {DATA_ROOT}")
    print(f"Free space     : {free_space_gb(DATA_ROOT):.1f} GB")
    print(f"HF_TOKEN       : {'set' if hf_token() else 'NOT SET'}")

    try:
        import datasets as hf_datasets  # noqa: F401
        print("datasets lib   : installed")
    except ImportError:
        print("datasets lib   : NOT installed  (pip install -r requirements-ml.txt)")

    ready, blocked, present = [], [], []
    for dataset in DATASETS.values():
        local = DATA_ROOT / dataset.key
        if local.exists() and any(local.iterdir()):
            present.append(dataset)
        elif dataset.access == Access.OPEN:
            ready.append(dataset)
        elif dataset.access == Access.HF_AUTH and hf_token():
            ready.append(dataset)
        else:
            blocked.append(dataset)

    if present:
        print("\nAlready downloaded:")
        for dataset in present:
            size = sum(f.stat().st_size for f in (DATA_ROOT / dataset.key).rglob("*")
                       if f.is_file()) / (1024 ** 3)
            print(f"  {dataset.key:<18} {size:.1f} GB")

    if ready:
        print("\nReady to download now:")
        for dataset in ready:
            print(f"  {dataset.key:<18} ~{human_gb(dataset.size_gb):>8}  {dataset.name}")

    if blocked:
        print("\nWaiting on you (account, licence acceptance, or purchase):")
        for dataset in blocked:
            reason = {
                Access.HF_AUTH: "needs HF_TOKEN + accepting terms on the dataset page",
                Access.REQUEST: "needs a request form",
                Access.PAID: "needs an LDC membership",
            }[dataset.access]
            print(f"  {dataset.key:<18} {reason}")

    print()
    return 0


def cmd_download(args) -> int:
    dataset = DATASETS.get(args.key)
    if dataset is None:
        print(f"Unknown dataset '{args.key}'.")
        return 1

    target = DATA_ROOT / dataset.key
    target.mkdir(parents=True, exist_ok=True)

    available = free_space_gb(DATA_ROOT)
    if available < dataset.size_gb * 1.3 and not args.force:
        print(f"Not enough disk space: {dataset.name} needs roughly "
              f"{human_gb(dataset.size_gb)} (plus room to extract) and "
              f"{available:.1f} GB is free.")
        print("Re-run with --force to try anyway.")
        return 1

    if dataset.access in (Access.REQUEST, Access.PAID):
        print(f"\n{dataset.name} cannot be fetched automatically.\n")
        print(_instructions(dataset))
        print()
        return 1

    if dataset.source == Source.HF:
        return _download_hf(dataset, target, args)
    if dataset.source == Source.URL:
        return _download_urls(dataset, target, args)

    print(f"\n{dataset.name} is a manual download.\n")
    print(_instructions(dataset))
    return 1


def _download_hf(dataset: Dataset, target: Path, args) -> int:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("The `datasets` library is required: pip install -r requirements-ml.txt")
        return 1

    token = hf_token()
    if dataset.access == Access.HF_AUTH and not token:
        print(f"\n{dataset.name} requires a Hugging Face token.\n")
        print(_instructions(dataset))
        return 1

    print(f"Downloading {dataset.name} -> {target}")
    print(f"  hf id : {dataset.hf_id}  config: {dataset.hf_config or '-'}")
    print(f"  size  : ~{human_gb(dataset.size_gb)}  (this will take a while)\n")

    try:
        load_dataset(
            dataset.hf_id,
            dataset.hf_config,
            split=args.split or dataset.hf_split,
            cache_dir=str(target),
            token=token,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"\nDownload failed: {exc}\n")
        if "gated" in str(exc).lower() or "403" in str(exc):
            print("This looks like an un-accepted licence. Open")
            print(f"  https://huggingface.co/datasets/{dataset.hf_id}")
            print("and accept the terms while signed in, then retry.")
        return 1

    print(f"\nDone. Cached under {target}")
    return 0


def _download_urls(dataset: Dataset, target: Path, args) -> int:
    urls = dataset.download_urls
    if args.subset:
        urls = [u for u in urls if args.subset in u]
        if not urls:
            print(f"No archive matches subset '{args.subset}'. Available:")
            for url in dataset.download_urls:
                print(f"  {url.rsplit('/', 1)[-1]}")
            return 1

    for url in urls:
        filename = url.rsplit("/", 1)[-1]
        archive = target / filename

        if archive.exists() and not args.force:
            print(f"{filename} already present, skipping download")
        else:
            print(f"Fetching {filename} ...")
            try:
                _download_with_progress(url, archive)
            except Exception as exc:
                print(f"  failed: {exc}")
                return 1

        if not args.no_extract:
            _extract(archive, target)

    print(f"\nDone. Files under {target}")
    return 0


def _download_with_progress(url: str, destination: Path) -> None:
    started = time.time()

    def hook(count: int, block_size: int, total_size: int) -> None:
        downloaded = count * block_size
        elapsed = max(time.time() - started, 0.001)
        speed = downloaded / elapsed / (1024 ** 2)
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(f"\r  {percent:5.1f}%  {downloaded / (1024**2):8.1f} MB  "
                  f"{speed:5.1f} MB/s", end="", flush=True)
        else:
            print(f"\r  {downloaded / (1024**2):8.1f} MB  {speed:5.1f} MB/s",
                  end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=hook)
    print()


def _extract(archive: Path, target: Path) -> None:
    if not archive.exists():
        return
    print(f"  extracting {archive.name} ...")
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
        elif ".tar" in archive.suffixes or archive.suffix in (".tgz", ".gz"):
            with tarfile.open(archive) as handle:
                # filter="data" refuses absolute paths and traversal entries.
                handle.extractall(target, filter="data")
        else:
            print(f"  unknown archive type, leaving {archive.name} as-is")
            return
        print("  extracted")
    except Exception as exc:
        print(f"  extraction failed: {exc}")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="SpeakTwin dataset manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list every known dataset")
    p_list.add_argument("--task", help="filter by task, e.g. filler_detection")
    p_list.add_argument("--open-only", action="store_true",
                        help="only those needing no account")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="details and access instructions")
    p_info.add_argument("key")
    p_info.set_defaults(func=cmd_info)

    p_check = sub.add_parser("check", help="what is present, ready, or blocked")
    p_check.set_defaults(func=cmd_check)

    p_get = sub.add_parser("download", help="download a dataset")
    p_get.add_argument("key")
    p_get.add_argument("--split", help="HF split, e.g. train / validation")
    p_get.add_argument("--subset", help="substring of the archive to fetch")
    p_get.add_argument("--no-extract", action="store_true")
    p_get.add_argument("--force", action="store_true",
                       help="re-download, and ignore the disk space check")
    p_get.set_defaults(func=cmd_download)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
