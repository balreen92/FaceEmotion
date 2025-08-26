import os, glob, json, numpy as np, cv2 as cv
from collections import defaultdict

FER_ONNX = "facial_expression_recognition_mobilefacenet_2022july.onnx"
YUNET_ONNX = "face_detection_yunet_2023mar.onnx"
LABELS = [l.strip() for l in open("labels.txt")]
IMG_DIR = "/Users/balreenkaur/Documents/fer_model/train"   # change if needed

fer = cv.dnn.readNet(FER_ONNX)

def make_detector(w, h):
    return cv.FaceDetectorYN_create(YUNET_ONNX, "", (w,h), 0.9, 0.3, 5000)

def predict(img_bgr, det=None):
    h, w = img_bgr.shape[:2]
    if det is None: det = make_detector(w, h)
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    if faces is None or len(faces)==0:
        return None, None
    x,y,ww,hh = faces[0][:4].astype(int)
    crop = img_bgr[max(0,y):y+hh, max(0,x):x+ww]
    if crop.size==0: return None, None
    blob = cv.dnn.blobFromImage(crop, 1/255.0, (112,112), (0,0,0), swapRB=True, crop=True)
    fer.setInput(blob)
    prob = fer.forward().squeeze()
    return int(np.argmax(prob)), prob

tot, correct = 0, 0
det = None
per_class = defaultdict(lambda: {"tp":0,"n":0})
confmat = np.zeros((7,7), dtype=int)

for ci, cls in enumerate(LABELS):
    paths=[]
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        paths += glob.glob(os.path.join(IMG_DIR, cls, ext))
    for p in paths:
        img = cv.imread(p)
        if img is None: continue
        if det is None: det = make_detector(img.shape[1], img.shape[0])
        pred, _ = predict(img, det)
        if pred is None: continue
        confmat[ci, pred]+=1
        per_class[cls]["n"]+=1
        tot += 1
        if pred==ci:
            per_class[cls]["tp"]+=1
            correct += 1

overall = (correct/max(1,tot))*100.0
by_class = {c:(v["tp"]/v["n"]*100.0 if v["n"] else None) for c,v in per_class.items()}

print(f"Overall accuracy: {overall:.2f}% on {tot} images")
print("Per-class (%):", json.dumps(by_class, indent=2))
print("Confusion matrix (rows=true, cols=pred):\n", confmat)
