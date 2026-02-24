from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sounddevice as sd
import numpy as np
import librosa

app = FastAPI()

# serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

samplerate = 16000
duration = 1

@app.get("/")
def root():
    return FileResponse("static/index.html")

def record_chunk():
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=1,
                   dtype="float32")
    sd.wait()
    return audio.flatten()

def estimate_pitch(y):
    pitches, magnitudes = librosa.piptrack(y=y, sr=samplerate)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    return float(np.mean(pitch)) if len(pitch) else 0.0

@app.get("/analyze")
def analyze():
    y = record_chunk()

    energy = float(np.sqrt(np.mean(y**2)))
    pitch = estimate_pitch(y)

    if energy < 0.01:
        msg = "No speech detected"
    elif pitch < 120:
        msg = "Tone low / monotone"
    elif pitch > 300:
        msg = "Pitch high / tense"
    else:
        msg = "Good speaking tone"

    return JSONResponse({"message": msg, "pitch": pitch, "energy": energy})