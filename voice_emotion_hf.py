import argparse, os, sys, time, queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from transformers import pipeline
import torch

SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_ID = "superb/hubert-large-superb-er"  # SER model on HF

def record_wav(seconds=10.0, out_path="recordings/recording.wav"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print("Ohh", status, file=sys.stderr)
        q.put(indata.copy())

    print(f"Recording {seconds}s at {SAMPLE_RATE} Hz…")
    chunks = []
    collected = 0
    frames_target = int(seconds * SAMPLE_RATE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=cb):
        t0 = time.time()
        while collected < frames_target:
            chunk = q.get()
            chunks.append(chunk)
            collected += chunk.shape[0]

    audio = np.concatenate(chunks, axis=0).astype(np.float32).squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(out_path, audio, SAMPLE_RATE)
    print(f"Saved: {out_path} (≈{time.time()-t0:.1f}s)")
    return out_path

def load_pipeline():
    device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else -1)
    # Note: HF pipeline on mac can use CPU or MPS indirectly; -1 means CPU
    print("Loading model:", MODEL_ID, "| device:", "CPU/MPS" if device == -1 else "CUDA")
    clf = pipeline("audio-classification", model=MODEL_ID)  # uses torchaudio to load wav
    return clf

def pretty(results, top_k=5):
    print("Top predictions:")
    for r in results[:top_k]:
        print(f"  - {r['label']:<20} {r['score']*100:5.1f}%")

def main():
    ap = argparse.ArgumentParser(description="Voice Emotion Detection (HF pipeline)")
    ap.add_argument("--file", type=str, default=None, help="Path to a WAV/FLAC/MP3 file.")
    ap.add_argument("--seconds", type=float, default=10.0, help="If no --file, record this many seconds.")
    ap.add_argument("--save", type=str, default="recordings/recording.wav", help="Where to save recorded audio.")
    args = ap.parse_args()

    if args.file is None:
        wav_path = record_wav(args.seconds, args.save)
    else:
        wav_path = args.file
        if not os.path.isfile(wav_path):
            print(f"file not found: {wav_path}")
            sys.exit(1)

    clf = load_pipeline()
    results = clf(wav_path)  # returns list of dicts: [{'label': '...', 'score': 0.x}, ...]
    # Some models return a single dict; normalize to list:
    if isinstance(results, dict):
        results = [results]

    # Sort high→low just in case
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    print("\nPredicted emotion:", results[0]["label"])
    pretty(results, top_k=min(5, len(results)))

if __name__ == "__main__":
    main()
