import sounddevice as sd
import numpy as np
import librosa
import queue
import time

samplerate = 16000
duration = 1  # seconds per chunk
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def estimate_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch) > 0:
        return float(np.mean(pitch))
    return 0.0

def speech_energy(y):
    return float(np.sqrt(np.mean(y**2)))

def analyze_chunk(chunk):
    y = chunk.flatten()

    # volume
    energy = speech_energy(y)

    # pitch
    pitch = estimate_pitch(y, samplerate)

    # crude speaking detection
    speaking = energy > 0.01

    return {
        "energy": energy,
        "pitch": pitch,
        "speaking": speaking
    }

def run_audio_loop():
    print("Listening... Speak now")

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        callback=audio_callback,
        blocksize=int(samplerate * duration),
    ):
        while True:
            chunk = audio_queue.get()
            result = analyze_chunk(chunk)

            if result["speaking"]:
                if result["pitch"] < 120:
                    print("⚠️ Tone low / possibly monotone")
                elif result["pitch"] > 300:
                    print("⚠️ Pitch high / possibly tense")
                else:
                    print("✅ Speaking normally")

            time.sleep(0.1)

if __name__ == "__main__":
    run_audio_loop()