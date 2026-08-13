# Datasets & Model Training

Everything SpeakTwin can train on, what it costs, and what you have to do
by hand.

```bash
python scripts/datasets.py list          # everything
python scripts/datasets.py check         # what's ready vs. blocked on you
python scripts/datasets.py info <key>    # details + access steps
python scripts/datasets.py download <key>
```

---

## What has to be done by a human

Four of these corpora are behind a click-through licence agreement. Accepting
a licence is a legal act performed by a named person, so it can't be scripted,
delegated, or done from a terminal — the account and the agreement have to be
yours. `scripts/datasets.py check` prints exactly which ones are waiting.

Everything downstream of that acceptance **is** automated: once `HF_TOKEN` is
in your `.env`, `download` handles the rest.

---

## Start here

| Priority | Dataset | Why | Access |
|---|---|---|---|
| 1 | **PodcastFillers** | The only corpus built for filler detection. Timed `um`/`uh` events. | Request form |
| 2 | **speechocean762** | Human delivery ratings under a fully open licence. 409 MB. | Open ✅ |
| 3 | **AMI Meeting** | Natural disfluent speech, CC BY (commercial-safe). | HF login |
| 4 | **TED-LIUM 3** | 450 h of actual public speaking. | HF login |

### Hugging Face setup (unlocks AMI, TED-LIUM, Common Voice, VoxPopuli)

1. Create a free account — <https://huggingface.co/join>
2. Open each dataset page and click **Agree and access repository**:
   - <https://huggingface.co/datasets/edinburghcstr/ami>
   - <https://huggingface.co/datasets/LIUM/tedlium>
   - <https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0>
   - <https://huggingface.co/datasets/facebook/voxpopuli>
3. Create a **read** token — <https://huggingface.co/settings/tokens>
4. Add it to `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
5. `python scripts/datasets.py download ami`

The same token unlocks the gated **pyannote** diarization model. That one
additionally needs you to accept terms at
<https://hf.co/pyannote/speaker-diarization-3.1>.

### PodcastFillers

1. Request access at <https://podcastfillers.github.io/>
2. Extract the release into `data/podcastfillers/`
3. `python training/train_filler_detector.py --data-dir data/podcastfillers --build-manifest`

Licensed for **non-commercial research**. Check the terms before shipping
SpeakTwin commercially — CC BY corpora like AMI are the safer base if that's
the plan.

---

## Licence summary

Licence drives what you can ship, so it's worth reading before you train.

| Dataset | Licence | Commercial use |
|---|---|---|
| Common Voice, VoxPopuli | CC0 | ✅ Unrestricted |
| speechocean762, AMI, LibriSpeech | CC BY 4.0 | ✅ With attribution |
| RAVDESS | CC BY-NC-SA | ❌ Non-commercial |
| PodcastFillers | Research only | ❌ Non-commercial |
| TED-LIUM 3 | CC BY-NC-ND | ❌ Non-commercial, no derivatives |
| Switchboard, Fisher | LDC | Per your membership |
| IEMOCAP, MSP-Podcast | Academic agreement | ❌ Research only |

A model's output generally inherits its training data's restrictions. If
SpeakTwin is ever commercial, build on **CC0 / CC BY** corpora only.

---

## Training

### Acoustic filler detector — the one worth training

Whisper deletes disfluencies. It was trained largely on cleaned transcripts,
and enabling its VAD filter trims the hesitation regions as well. So
transcript-based filler counting under-counts structurally, and no regex
improvement fixes it — the evidence only survives in the waveform.

```bash
python training/train_filler_detector.py \
    --data-dir data/podcastfillers \
    --build-manifest \
    --output-dir models/filler-detector \
    --backbone microsoft/wavlm-base-plus \
    --epochs 3 --fp16
```

Then:
```env
ML_DISFLUENCY_ENABLED=true
ML_DISFLUENCY_MODEL=./models/filler-detector
```

The backend merges acoustic and text counts by taking the **maximum** per
label, not the sum — when Whisper does keep an "um", both detectors see the
same event and adding them would double-count it.

Frame labels are overwhelmingly the negative class, so the script reports
positive-class precision/recall/F1 alongside accuracy. Accuracy on its own
will look excellent while the model predicts "none" for everything.

### Learned confidence score — runnable now

[confidence_score.py](backend/services/confidence_score.py) is a hand-weighted
average (WPM 25%, pitch 25%, energy 20%, fillers 30%). Those weights are
guesses. **speechocean762** has real human ratings, and it needs no account:

```bash
python scripts/datasets.py download speechocean762      # 409 MB
python training/train_confidence_scorer.py --target fluency
```

The script keeps the sub-score *functions* — they encode real domain knowledge
about what good delivery sounds like — and fits only the weights, so the result
drops straight back into config.

**It also refuses to endorse weights the data cannot support.** On
speechocean762 the energy sub-score has σ = 0.010 (studio recordings, all at
the same level) and 95% of clips contain zero fillers, so those two weights are
fitted on almost no signal. The script detects near-constant features, keeps
the existing values for them, and renormalises the rest — rather than printing
four confident-looking numbers.

Measured on 400 utterances: R² = 0.20, with `wpm` and `pitch_variation`
trustworthy and `energy` and `filler_usage` flagged unreliable.

> The corpus is *read* speech scored for pronunciation, not spontaneous public
> speaking. Treat the output as a better-grounded prior than the guess, not a
> finished answer. Weights fitted on rated public speaking would be materially
> better; no such corpus is openly available.

Set `ML_PROSODY_FULL_VECTOR=true` to emit all 88 eGeMAPS features per chunk if
you want a richer feature vector.

---

## Compute

No GPU is needed to *run* SpeakTwin — every enabled model works on CPU.
Training the filler detector on CPU is not practical.

| Option | Notes |
|---|---|
| **Kaggle Notebooks** | ~30 GPU-hours/week free. Best free tier for this size of fine-tune. |
| **Google Colab** | Free T4; Pro for A100/L4. |
| **RunPod / Vast.ai** | ~$0.20–0.60/hr. A full fine-tune is a few hours. |
| **Modal / Lightning AI** | Serverless GPU; same code for training and inference. |

---

## Disk

`scripts/datasets.py` refuses a download that won't fit (pass `--force` to
override) and `check` prints free space. Rough sizes: Common Voice ~80 GB,
TED-LIUM ~54 GB, LibriSpeech ~60 GB (`--subset dev-clean` is ~340 MB and
enough to smoke-test a pipeline), PodcastFillers ~40 GB, speechocean762
~409 MB.

Downloads land in `data/`, which is gitignored.
