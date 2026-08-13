"""
SpeakTwin - Speech Dataset Registry
====================================
Every corpus worth training a SpeakTwin model on, with what it costs you
in disk, licence terms, and manual effort.

Access reality, because it drives everything else:

  * `OPEN`      - download starts immediately, no account
  * `HF_AUTH`   - free Hugging Face account, plus clicking "agree" on the
                  dataset page. The click is a licence acceptance and
                  cannot be automated or done on someone's behalf.
  * `REQUEST`   - a form, an academic affiliation, or a signature. Days,
                  not minutes.
  * `PAID`      - LDC membership or per-corpus fee.

Sizes are approximate download sizes and are there to stop you starting a
40 GB pull on a laptop with 12 GB free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Access(str, Enum):
    OPEN = "open"
    HF_AUTH = "hf_auth"
    REQUEST = "request"
    PAID = "paid"


class Source(str, Enum):
    HF = "huggingface"
    URL = "direct_url"
    MANUAL = "manual"


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    task: str
    access: Access
    source: Source
    size_gb: float
    licence: str
    url: str
    description: str
    hf_id: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: Optional[str] = None
    download_urls: List[str] = field(default_factory=list)
    notes: str = ""
    recommended: bool = False


# ---------------------------------------------------------------------------
# Filler / disfluency detection - the models SpeakTwin most needs
# ---------------------------------------------------------------------------
DATASETS: Dict[str, Dataset] = {}


def _add(dataset: Dataset) -> None:
    DATASETS[dataset.key] = dataset


_add(Dataset(
    key="podcastfillers",
    name="PodcastFillers",
    task="filler_detection",
    access=Access.REQUEST,
    source=Source.MANUAL,
    size_gb=40.0,
    licence="Research use only (non-commercial)",
    url="https://podcastfillers.github.io/",
    hf_id=None,
    recommended=True,
    description=(
        "Podcast audio with filler words (um, uh) annotated as timed events, "
        "alongside laughter, breath, and speech. Purpose-built for exactly "
        "the acoustic filler detector SpeakTwin needs."
    ),
    notes=(
        "THE dataset for this project. Fill in the request form on the site; "
        "you get a download link. Non-commercial licence - check it before "
        "shipping SpeakTwin commercially."
    ),
))

_add(Dataset(
    key="switchboard",
    name="Switchboard-1 Release 2",
    task="disfluency",
    access=Access.PAID,
    source=Source.MANUAL,
    size_gb=25.0,
    licence="LDC (membership or purchase)",
    url="https://catalog.ldc.upenn.edu/LDC97S62",
    description=(
        "Conversational telephone speech; the canonical corpus for "
        "disfluency research, with annotated repairs, restarts, and edits."
    ),
    notes="Needs an LDC membership. The disfluency annotations are LDC99T42.",
))

_add(Dataset(
    key="buckeye",
    name="Buckeye Corpus",
    task="disfluency",
    access=Access.REQUEST,
    source=Source.MANUAL,
    size_gb=5.0,
    licence="Free for research; registration required",
    url="https://buckeyecorpus.osu.edu/",
    description=(
        "Spontaneous conversational speech with close phonetic labelling, "
        "including hesitations. Free, unlike Switchboard."
    ),
))

_add(Dataset(
    key="ami",
    name="AMI Meeting Corpus",
    task="disfluency,diarization",
    access=Access.HF_AUTH,
    source=Source.HF,
    size_gb=10.0,
    licence="CC BY 4.0",
    url="https://groups.inf.ed.ac.uk/ami/corpus/",
    hf_id="edinburghcstr/ami",
    hf_config="ihm",
    recommended=True,
    description=(
        "100 hours of multi-speaker meeting recordings. Naturally disfluent, "
        "permissively licensed, and doubles as diarization training data."
    ),
    notes="CC BY means commercial use is fine with attribution.",
))


# ---------------------------------------------------------------------------
# Public speaking / presentation style
# ---------------------------------------------------------------------------
_add(Dataset(
    key="tedlium",
    name="TED-LIUM 3",
    task="asr,public_speaking",
    access=Access.HF_AUTH,
    source=Source.HF,
    size_gb=54.0,
    licence="CC BY-NC-ND 3.0",
    url="https://www.openslr.org/51/",
    hf_id="LIUM/tedlium",
    hf_config="release3",
    recommended=True,
    description=(
        "~450 hours of TED talks with transcripts. The closest public corpus "
        "to SpeakTwin's actual use case: prepared, performed public speaking."
    ),
    notes="NC-ND licence - research and evaluation only, no derivatives.",
))

_add(Dataset(
    key="voxpopuli",
    name="VoxPopuli",
    task="asr,public_speaking",
    access=Access.HF_AUTH,
    source=Source.HF,
    size_gb=30.0,
    licence="CC0 (public domain)",
    url="https://github.com/facebookresearch/voxpopuli",
    hf_id="facebook/voxpopuli",
    hf_config="en",
    description=(
        "European Parliament speeches - formal, prepared oratory. CC0, so "
        "the least restrictive large corpus of public speaking."
    ),
))


# ---------------------------------------------------------------------------
# General ASR / self-supervised pre-training
# ---------------------------------------------------------------------------
_add(Dataset(
    key="commonvoice",
    name="Common Voice (v17, English)",
    task="asr",
    access=Access.HF_AUTH,
    source=Source.HF,
    size_gb=80.0,
    licence="CC0 (public domain)",
    url="https://commonvoice.mozilla.org/en/datasets",
    hf_id="mozilla-foundation/common_voice_17_0",
    hf_config="en",
    description=(
        "Crowd-sourced read speech, many accents and recording conditions. "
        "CC0, so usable commercially without restriction."
    ),
    notes="Read speech: almost no disfluencies. Wrong choice for filler models.",
))

_add(Dataset(
    key="librispeech",
    name="LibriSpeech",
    task="asr",
    access=Access.OPEN,
    source=Source.URL,
    size_gb=60.0,
    licence="CC BY 4.0",
    url="https://www.openslr.org/12/",
    hf_id="openslr/librispeech_asr",
    download_urls=[
        "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    ],
    description=(
        "1000 hours of read audiobooks. The standard ASR benchmark and the "
        "only large corpus here that downloads with no account at all."
    ),
    notes=(
        "Read speech from books - no fillers, no spontaneity. Good for "
        "sanity-checking a pipeline, not for training delivery models. "
        "dev-clean alone (~340 MB) is enough to smoke-test."
    ),
))


# ---------------------------------------------------------------------------
# Scored / rated speech - supervision for a learned confidence model
# ---------------------------------------------------------------------------
_add(Dataset(
    key="speechocean762",
    name="speechocean762",
    task="pronunciation_scoring",
    access=Access.OPEN,
    source=Source.URL,
    size_gb=0.4,
    licence="CC BY 4.0 (fully open)",
    url="https://www.openslr.org/101/",
    hf_id="mispeech/speechocean762",
    download_urls=["https://www.openslr.org/resources/101/speechocean762.tar.gz"],
    recommended=True,
    description=(
        "5000 utterances with *human* proficiency scores at sentence, word, "
        "and phoneme level: accuracy, fluency, prosody, completeness."
    ),
    notes=(
        "Rare and valuable: real human delivery ratings under an open "
        "licence. This is the supervision for replacing the hand-weighted "
        "confidence score with a learned one. Small enough to just fetch."
    ),
))


# ---------------------------------------------------------------------------
# Emotion / affect
# ---------------------------------------------------------------------------
_add(Dataset(
    key="ravdess",
    name="RAVDESS",
    task="emotion",
    access=Access.OPEN,
    source=Source.URL,
    size_gb=1.5,
    licence="CC BY-NC-SA 4.0",
    url="https://zenodo.org/record/1188976",
    download_urls=[
        "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    ],
    description=(
        "24 actors performing 8 emotions. Clean, balanced, downloads without "
        "an account - the easiest place to start on emotion."
    ),
    notes="Acted, not natural. Models trained on it transfer imperfectly.",
))

_add(Dataset(
    key="cremad",
    name="CREMA-D",
    task="emotion",
    access=Access.OPEN,
    source=Source.MANUAL,
    size_gb=2.0,
    licence="Open Database License",
    url="https://github.com/CheyneyComputerScience/CREMA-D",
    description="7442 acted clips, 91 actors, 6 emotions, crowd-validated.",
))

_add(Dataset(
    key="iemocap",
    name="IEMOCAP",
    task="emotion",
    access=Access.REQUEST,
    source=Source.MANUAL,
    size_gb=12.0,
    licence="Academic licence, request form",
    url="https://sail.usc.edu/iemocap/",
    description=(
        "Dyadic emotional interaction with audio, video, and motion capture. "
        "The standard emotion benchmark."
    ),
    notes="Request form; academic affiliation expected. Takes days.",
))

_add(Dataset(
    key="msp_podcast",
    name="MSP-Podcast",
    task="emotion",
    access=Access.REQUEST,
    source=Source.MANUAL,
    size_gb=30.0,
    licence="Academic licence agreement",
    url="https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html",
    description=(
        "Large-scale *natural* emotional speech harvested from podcasts - "
        "far more realistic than acted corpora."
    ),
))


# ---------------------------------------------------------------------------
# Speaker recognition / diarization
# ---------------------------------------------------------------------------
_add(Dataset(
    key="voxceleb1",
    name="VoxCeleb1",
    task="speaker",
    access=Access.REQUEST,
    source=Source.MANUAL,
    size_gb=35.0,
    licence="CC BY 4.0, registration required",
    url="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
    description=(
        "Interview speech from 1251 speakers. What ECAPA-TDNN speaker "
        "embeddings are trained on."
    ),
    notes="You likely never need this - use the pretrained SpeechBrain model.",
))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def by_task(task: str) -> List[Dataset]:
    return [d for d in DATASETS.values() if task in d.task]


def recommended() -> List[Dataset]:
    return [d for d in DATASETS.values() if d.recommended]


def downloadable_without_account() -> List[Dataset]:
    return [d for d in DATASETS.values() if d.access == Access.OPEN]
