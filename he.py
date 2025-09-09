python -m venv .venv
.venv\Scripts\activate
python evaluate_hf.py --root C:\Users\<YourName>\audb\emodb\2.0.0\d3b62a9b


python -m pip install --upgrade pip
pip install "numpy<2.0" scipy==1.13.1
pip install torch==2.2.2+cpu torchaudio==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.2 sounddevice==0.4.7 soundfile==0.13.1 tqdm==4.67.1
pip install audb  # if you want EmoDB


Download from: https://www.gyan.dev/ffmpeg/builds/

(pick release full → unzip somewhere, e.g., C:\ffmpeg)

Add C:\ffmpeg\bin to your PATH:

Search Environment Variables → Edit environment variables

Edit Path → Add C:\ffmpeg\bin

Restart PyCharm / terminal




import os, glob, argparse
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import pipeline

MODEL_ID = "superb/hubert-large-superb-er"
TARGET_SR = 16000

def load_audio_noffmpeg(path, target_sr=TARGET_SR):
    audio, sr = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)
        sr = target_sr
    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr

def iter_files(root):
    for label in os.listdir(root):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.wav")):
            yield label.lower(), p

def main():
    ap = argparse.ArgumentParser(description="Evaluate SER (no ffmpeg)")
    ap.add_argument("--root", required=True, help="Dataset root with subfolders per label.")
    args = ap.parse_args()

    clf = pipeline("audio-classification", model=MODEL_ID)

    total = correct = 0
    per = {}

    for true_lbl, path in iter_files(args.root):
        audio, sr = load_audio_noffmpeg(path, TARGET_SR)
        res = clf({"array": audio, "sampling_rate": sr})
        if isinstance(res, dict): res = [res]
        res.sort(key=lambda x: x["score"], reverse=True)
        pred = res[0]["label"].lower()

        total += 1
        per.setdefault(true_lbl, {"n":0, "tp":0})
        per[true_lbl]["n"] += 1

        # loose match (handles 'happy' vs 'happiness', etc.)
        hit = (true_lbl in pred) or (pred in true_lbl)
        if hit:
            correct += 1
            per[true_lbl]["tp"] += 1

    if total == 0:
        print("No .wav files found under", args.root); return

    print(f"\n✅ Overall accuracy: {correct/total*100:.2f}%  ({correct}/{total})")
    for lbl, s in per.items():
        n, tp = s["n"], s["tp"]
        acc = (tp/n*100) if n else 0.0
        print(f"  {lbl:15s}: {acc:5.1f}%  ({tp}/{n})")

if __name__ == "__main__":
    main()

