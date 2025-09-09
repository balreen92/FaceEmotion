# save as evaluate_hf.py
import audb
audb.load("emodb")

import os, glob, argparse
from transformers import pipeline

MODEL_ID = "superb/hubert-large-superb-er"

def iter_files(root):
    for label in os.listdir(root):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.wav")):
            yield label.lower(), p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root/anger/*.wav, root/happy/*.wav, etc.")
    args = ap.parse_args()

    clf = pipeline("audio-classification", model=MODEL_ID)
    total = correct = 0
    per = {}

    for true_lbl, path in iter_files(args.root):
        res = clf(path)
        if isinstance(res, dict): res = [res]
        res.sort(key=lambda x: x["score"], reverse=True)
        pred = res[0]["label"].lower()
        total += 1
        per.setdefault(true_lbl, {"n":0, "tp":0})
        per[true_lbl]["n"] += 1
        if pred.startswith(true_lbl) or true_lbl in pred:
            correct += 1
            per[true_lbl]["tp"] += 1

    if total == 0:
        print("No files found.")
        return
    print(f"✅ Overall accuracy: {correct/total*100:.2f}%  ({correct}/{total})")
    for k,v in per.items():
        n, tp = v["n"], v["tp"]
        print(f"  {k:12s}: {tp/n*100 if n else 0:.1f}%  ({tp}/{n})")

if __name__ == "__main__":
    main()
