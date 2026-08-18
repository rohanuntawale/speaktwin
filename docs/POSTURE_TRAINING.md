# Posture data, training, and launch

## Launch the platform

From PowerShell in the repository root:

```powershell
cd C:\Users\Admin\AI-MIRROR
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000). Camera permissions work
on `localhost` or HTTPS. The API health check is:

```powershell
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing
```

To stop the server, press `Ctrl+C` in the terminal running Uvicorn.

## Download the posture data

The downloader fetches the three public Kaggle datasets selected for this
project and writes `data/posture/manifest.json` with the source and license:

```powershell
python scripts/download_posture_data.py
```

Individual downloads can be resumed/reused:

```powershell
python scripts/download_posture_data.py --dataset cctv
python scripts/download_posture_data.py --dataset keypoints
python scripts/download_posture_data.py --dataset silhouettes
python scripts/download_posture_data.py --dataset wlu
```

The CCTV archive currently contains images but no label files or class
directories. It is therefore retained as real visual reference data, not
silently treated as supervised correct/incorrect labels.

## Train the synthetic prototype

The live application uses MediaPipe landmark-derived features. The prototype
trainer creates controlled landmark-feature perturbations for `correct`,
`sunk`, `forward_head`, `leaning`, and `uncertain`:

```powershell
python training/train_posture_model.py --synthetic-only
```

Outputs:

```text
models/posture/posture_classifier.joblib
models/posture/metrics.json
```

## Train the requested 50/50 model

Real data must be labelled in the same landmark-feature format before it can
be mixed with synthetic data. Copy the template first, then replace its
example rows with real labelled sessions. The example rows are only a format
reference and are intentionally rejected by the trainer; collect at least 20
rows per class from at least 3 people:

```powershell
Copy-Item data\posture\real_landmarks.template.csv data\posture\real_landmarks.csv
```

The CSV columns are:

```text
label,subject,split,shoulder_tilt,head_tilt,torso_lean,openness,head_scale,head_offset,forward_head,hand_face_dist,head_drop
```

Then run:

```powershell
python training/train_posture_model.py --real-csv data/posture/real_landmarks.csv
```

To create compatible real rows from the downloaded WLU videos, use the
labelled `Sit To Stand` subset:

```powershell
python training\extract_wlu_landmarks.py
python training\train_posture_model.py --real-csv data\posture\real_landmarks.csv
```

The extractor maps WLU correct videos to `correct` and incorrect videos to
`uncertain`; it deliberately does not call rehabilitation errors `sunk` or
`forward_head` without those labels.

The trainer balances the synthetic and real sample counts, splits by subject,
and writes a classification report and confusion matrix. Do not use random
frame-level splitting; adjacent frames from one person would leak into both
training and validation.

## Useful checks

```powershell
python -m pytest -q
node --check frontend\pose.js
node --check frontend\app.js
```
