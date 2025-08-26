import argparse, os, glob, json, numpy as np, cv2 as cv
from collections import Counter
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

LABELS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)  # path to test_set/
    ap.add_argument("--yunet", required=True)  # face_detection_yunet_2023mar.onnx
    ap.add_argument("--pad", type=float, default=0.25)
    args = ap.parse_args()

    # Init YuNet detector
    det = cv.FaceDetectorYN_create(args.yunet, "", (320, 240), 0.9, 0.3, 5000)
    # Init HSEmotion
    fer = HSEmotionRecognizer(model_name="enet_b2_7")

    def crop112(img):
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        if faces is None or len(faces) == 0: return None
        x, y, wf, hf = faces[0][:4].astype(int)
        crop = img[max(0, y):y + hf, max(0, x):x + wf]
        if crop.size == 0: return None
        return cv.resize(crop, (112, 112), interpolation=cv.INTER_AREA)

    exts = ("*.jpg", "*.jpeg", "*.png")
    conf = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    totals = Counter();
    correct = 0;
    total = 0;
    noface = 0

    for i, lab in enumerate(LABELS):
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(args.root, lab, e)))
        for f in files:
            img = cv.imread(f)
            if img is None: continue
            face = crop112(img)
            if face is None:
                noface += 1
                continue
            emo, probs = fer.predict_emotions(face, logits=False)
            # Map HSEmotion output to our label set
            MAP = {"Happiness": "happy", "Sadness": "sad", "Fear": "fearful",
                   "Surprise": "surprised", "Anger": "angry", "Disgust": "disgust", "Neutral": "neutral"}
            pred_lab = MAP.get(emo, emo.lower())
            pred_idx = LABELS.index(pred_lab)
            conf[i, pred_idx] += 1
            totals[lab] += 1
            correct += int(pred_idx == i);
            total += 1

    overall = (correct / max(1, total)) * 100.0
    per_class = {LABELS[i]: (conf[i, i] / max(1, conf[i].sum()) * 100.0) for i in range(len(LABELS))}
    macro = float(np.mean(list(per_class.values()))) if total > 0 else 0.0

    print(json.dumps({
        "overall_accuracy_percent": round(overall, 2),
        "macro_accuracy_percent": round(macro, 2),
        "samples_used": int(total),
        "skipped_no_face": int(noface),
        "per_class_percent": {k: round(v, 2) for k, v in per_class.items()}
    }, indent=2))
    print("\nConfusion matrix (rows=true, cols=pred, order):", LABELS)
    print(conf)


if __name__ == "__main__":
    main()


