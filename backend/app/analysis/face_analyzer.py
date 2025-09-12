import cv2 as cv
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

LABELS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

class YuNet:
    def __init__(self, path, pad=0.30):
        self.det = cv.FaceDetectorYN_create(path, "", (320, 240), 0.9, 0.3, 5000)
        self.pad = float(pad)

    def crop112(self, bgr):
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0: return None
        x, y, ww, hh = [float(v) for v in faces[0][:4]]
        cx, cy = x + ww/2.0, y + hh/2.0
        side = max(ww, hh) * (1.0 + 2*self.pad)
        nx = int(max(0, cx - side/2.0)); ny = int(max(0, cy - side/2.0))
        nxe = int(min(w, cx + side/2.0)); nye = int(min(h, cy + side/2.0))
        crop = bgr[ny:nye, nx:nxe]
        if crop.size == 0: return None
        face = cv.resize(crop, (112, 112), interpolation=cv.INTER_AREA)
        return face

# Reuse your YuNet + HSEmotionRecognizer approach from the live FER script.
# (Same 7-class AffectNet head.) :contentReference[oaicite:3]{index=3}
_yunet = None
_hs = None

def load_face_models(yunet_path: str):
    global _yunet, _hs
    if _yunet is None: _yunet = YuNet(yunet_path, pad=0.30)
    if _hs is None: _hs = HSEmotionRecognizer(model_name="enet_b2_7")
    return _yunet, _hs

def classify_faces_from_frames(frames, yunet_path: str):
    yunet, hs = load_face_models(yunet_path)
    probs_accum = np.zeros(len(LABELS), dtype=np.float64)
    n = 0
    for f in frames:
        face = yunet.crop112(f)
        if face is None: continue
        _, probs = hs.predict_emotions(face, logits=False)  # probs aligned to LABELS
        probs_accum += np.array(probs, dtype=np.float64)
        n += 1
    if n == 0:
        return {"label": "neutral", "score": 0.0, "detail": []}
    avg = probs_accum / n
    idx = int(np.argmax(avg))
    return {
        "label": LABELS[idx],
        "score": float(avg[idx]),
        "detail": [{"label": LABELS[i], "score": float(avg[i])} for i in range(len(LABELS))]
    }
