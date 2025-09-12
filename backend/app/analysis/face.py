# face_accelerated.py
import json, math, cv2 as cv, numpy as np, onnxruntime as ort

# ---- Config (tune here) ----
ONNX_PATH   = r"..\..\models\facial_expression_recognition_mobilefacenet_2022july.onnx"
YUNET_PATH  = r"..\..\models\face_detection_yunet_2023mar.onnx"
LABELS_PATH = r"..\..\models\labels.json"  # list or dict
BIAS_JSON   = r"..\..\models\logit_bias_calibration.json"  # created in step 3
PAD = 0.20
SCORE_THR = 0.6     # YuNet confidence
NMS_THR   = 0.3
BLUR_VAR_MIN = 40.0 # variance of Laplacian
BRIGHT_MIN   = 18.0 # mean intensity lower bound
USE_TTA = True
EMA_ALPHA = 0.25    # 0.2–0.35 is good
TEMP = 0.9          # <1.0 sharpens probs a bit

def _load_labels(path):
    with open(path, "r") as f:
        L = json.load(f)
    if isinstance(L, dict):
        idx_to_label = {int(v): k for k, v in L.items()}
        labels = [idx_to_label[i] for i in sorted(idx_to_label)]
    else:
        labels = list(L)
    return labels

def _load_bias(path, n):
    try:
        with open(path, "r") as f:
            d = json.load(f)
        v = np.array([float(d.get(str(i), 0.0)) for i in range(n)], dtype=np.float32)
        return v
    except Exception:
        return np.zeros((n,), dtype=np.float32)

class YuNet:
    def __init__(self, path, pad=PAD, score=SCORE_THR, nms=NMS_THR, topk=5000):
        self.det = cv.FaceDetectorYN_create(path, "", (320,240), score, nms, topk)
        self.pad = float(pad)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        return faces

    def aligned_crop(self, bgr, out_size=(112,112)):
        faces = self.detect(bgr)
        if faces is None or len(faces)==0: return None
        f = faces[0]
        x,y,w,h = [float(v) for v in f[:4]]
        # eye landmarks (YuNet order: l0=right eye? we use first two)
        ex1,ey1, ex2,ey2 = [float(v) for v in f[5:9]]
        eye_c = ((ex1+ex2)/2.0, (ey1+ey2)/2.0)
        angle = math.degrees(math.atan2((ey2-ey1),(ex2-ex1)))
        M = cv.getRotationMatrix2D(eye_c, angle, 1.0)
        rot = cv.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), flags=cv.INTER_LINEAR)
        # padded square crop around original box (after rotation)
        side = max(w,h)*(1+2*self.pad)
        cx, cy = x+w/2.0, y+h/2.0
        nx = int(max(0, cx - side/2.0)); ny = int(max(0, cy - side/2.0))
        ex = int(min(rot.shape[1], cx + side/2.0)); ey = int(min(rot.shape[0], cy + side/2.0))
        crop = rot[ny:ey, nx:ex]
        if crop.size==0: return None
        face = cv.resize(crop, out_size, interpolation=cv.INTER_AREA)
        # color -> RGB, normalize, light enhancement
        face = cv.cvtColor(face, cv.COLOR_BGR2RGB).astype(np.float32)/255.0
        # CLAHE on L channel (LAB) for mild contrast boost
        lab = cv.cvtColor((face*255).astype(np.uint8), cv.COLOR_RGB2LAB)
        l,a,b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        l = clahe.apply(l)
        lab = cv.merge([l,a,b])
        face = cv.cvtColor(lab, cv.COLOR_LAB2RGB).astype(np.float32)/255.0
        # gentle gamma
        gamma = 0.9
        face = np.clip(face**gamma, 0.0, 1.0)
        return (face*255).astype(np.uint8)

def _lap_var(gray): return float(cv.Laplacian(gray, cv.CV_64F).var())

class ONNXFER:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        ishape = self.sess.get_inputs()[0].shape
        self.h = int(ishape[2] if isinstance(ishape[2], (int,np.integer)) else 112)
        self.w = int(ishape[3] if isinstance(ishape[3], (int,np.integer)) else 112)
        self.input_name = self.sess.get_inputs()[0].name

    def _pre(self, rgb):
        x = cv.resize(rgb, (self.w, self.h), interpolation=cv.INTER_AREA).astype(np.float32)/255.0
        x = np.transpose(x, (2,0,1))[None,...]  # NCHW
        return x

    def logits(self, rgb_uint8):
        x = self._pre(rgb_uint8)
        out = self.sess.run(None, {self.input_name: x})[0]
        return out[0].astype(np.float32)

class FaceEmotionAnalyzer:
    def __init__(self):
        self.labels = _load_labels(LABELS_PATH)
        self.bias   = _load_bias(BIAS_JSON, len(self.labels))
        self.det    = YuNet(YUNET_PATH)
        self.net    = ONNXFER(ONNX_PATH)
        self.ema    = None

    def _softmax_t(self, z):
        # temperature & numerical stability
        z = (z / max(TEMP, 1e-6)).astype(np.float32)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _update_ema(self, logit):
        if self.ema is None: self.ema = logit.copy()
        else: self.ema = (1-EMA_ALPHA)*self.ema + EMA_ALPHA*logit
        return self.ema

    def classify_frames(self, frames_bgr):
        accum = np.zeros((len(self.labels),), dtype=np.float64); n=0
        for f in frames_bgr:
            gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
            if _lap_var(gray) < BLUR_VAR_MIN: continue
            if gray.mean() < BRIGHT_MIN:      continue
            face = self.det.aligned_crop(f, (self.net.w, self.net.h))
            if face is None: continue
            # base logits (+ TTA)
            logit = self.net.logits(face)
            if USE_TTA:
                flip = cv.flip(face, 1)
                logit = 0.5*(logit + self.net.logits(flip))
            # calibration + EMA
            logit = logit + self.bias
            logit = self._update_ema(logit)
            probs = self._softmax_t(logit)
            accum += probs; n+=1

        if n==0: return {"label":"neutral","score":0.0,"detail":[]}
        avg = accum/n; idx = int(np.argmax(avg))
        return {"label": self.labels[idx],
                "score": float(avg[idx]),
                "detail":[{"label": self.labels[i], "score": float(avg[i])} for i in range(len(self.labels))]}
