
#!/usr/bin/env python3

"""
Evaluate an ONNX FER model on a folder tree with optional YuNet cropping and TTA.

Usage:
  python fer_eval.py --onnx /path/to/model.onnx --labels /path/to/labels.json --img_dir /path/to/test \
    --yunet face_detection_yunet_2023mar.onnx --pad 0.25 --align --tta

Outputs overall accuracy, macro accuracy, per-class metrics, and confusion matrix.
"""
import argparse, os, glob, json, math
from collections import defaultdict, Counter
import numpy as np
import onnxruntime as ort
import cv2 as cv

def load_labels(path):
    # expects dict: { "0":"angry", "1":"disgust", ... } or inverse; handle both
    with open(path, "r") as f:
        obj = json.load(f)
    labels = None
    if isinstance(obj, dict):
        # build index->label
        if all(k.isdigit() for k in obj.keys()):
            labels = [obj[str(i)] for i in range(len(obj))]
        else:
            # assume label->index
            items = sorted(((int(v),k) for k,v in obj.items()), key=lambda x:x[0])
            labels = [k for _,k in items]
    else:
        labels = list(obj)
    return labels

class YuNet:
    def __init__(self, onnx_path, pad=0.25, align=False):
        self.pad = pad
        self.align = align
        self.det = cv.FaceDetectorYN_create(onnx_path, "", (320,240), 0.9, 0.3, 5000)
    def crop112(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img_bgr)
        if faces is None or len(faces)==0: return None
        f = faces[0]
        x, y, ww, hh = [float(v) for v in f[:4]]
        le = (float(f[4]),  float(f[5]))
        re = (float(f[6]),  float(f[7]))
        cx, cy = x+ww/2.0, y+hh/2.0
        side = max(ww, hh)*(1.0+2*self.pad)
        nx = int(max(0, cx - side/2.0)); ny = int(max(0, cy - side/2.0))
        nxe = int(min(w, cx + side/2.0)); nye = int(min(h, cy + side/2.0))
        crop = img_bgr[ny:nye, nx:nxe]
        if crop.size==0: return None
        if self.align:
            dy = re[1]-le[1]; dx = re[0]-le[0]
            angle = -np.degrees(np.arctan2(dy, dx))
            ch, cw = crop.shape[:2]
            M = cv.getRotationMatrix2D((cw/2.0, ch/2.0), angle, 1.0)
            crop = cv.warpAffine(crop, M, (cw, ch), flags=cv.INTER_LINEAR)
        return cv.resize(crop, (112,112), interpolation=cv.INTER_AREA)

def preprocess(bgr):
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))[None, ...]  # 1x3x112x112
    return x

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--yunet", default=None)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    cropper = YuNet(args.yunet, pad=args.pad, align=args.align) if args.yunet else None

    # Collect images by class
    classes = [d for d in os.listdir(args.img_dir) if os.path.isdir(os.path.join(args.img_dir, d))]
    classes = sorted([c for c in classes if c in labels])
    idx = {c:i for i,c in enumerate(labels)}

    confmat = np.zeros((len(labels), len(labels)), dtype=np.int64)
    totals = Counter(); corrects = Counter()

    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    for c in classes:
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(args.img_dir, c, e))
        for p in paths:
            img = cv.imread(p); 
            if img is None: continue
            if cropper:
                face = cropper.crop112(img)
                if face is None:
                    continue  # skip no-face to separate detection from classification
            else:
                face = cv.resize(img, (112,112), interpolation=cv.INTER_AREA)

            x = preprocess(face)
            logits = sess.run([output_name], {input_name: x})[0]
            if args.tta:
                xflip = np.flip(x, axis=3).copy()
                logits2 = sess.run([output_name], {input_name: xflip})[0]
                logits = (logits + logits2) / 2.0
            pred = int(np.argmax(logits, axis=1)[0])
            gt = idx[c]
            confmat[gt, pred] += 1
            totals[c] += 1
            if pred == gt: corrects[c] += 1

    overall_n = int(sum(totals.values()))
    overall_acc = float(sum(corrects.values())) / overall_n * 100.0 if overall_n else 0.0
    per_class = {c: {"support": int(totals[c]), "accuracy": (float(corrects[c])/totals[c]*100.0 if totals[c] else None)} for c in labels}
    macro = np.mean([v["accuracy"] for v in per_class.values() if v["accuracy"] is not None]) if overall_n else 0.0

    print(json.dumps({
        "overall_accuracy_percent": round(overall_acc, 2),
        "macro_accuracy_percent": round(macro, 2),
        "samples": overall_n,
        "per_class": per_class
    }, indent=2))
    print("\nConfusion matrix (rows=true, cols=pred, order):", labels)
    print(confmat)
if __name__ == "__main__":
    main()
