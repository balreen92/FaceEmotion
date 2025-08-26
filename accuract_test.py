python run_fer.py --onnx model.onnx --labels labels.json --video demo.mp4 --yunet face_detection_yunet_2023mar.onnx --save_vis --display


import argparse, json, os, time, math, sys
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2 as cv
import onnxruntime as ort
from tqdm import tqdm

def log(msg: str): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        if all(k.isdigit() for k in obj.keys()):
            return [obj[str(i)] for i in range(len(obj))]
        else:
            items = sorted(((int(v), k) for k, v in obj.items()), key=lambda x: x[0])
            return [k for _, k in items]
    raise ValueError("labels.json format unsupported")

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

class YuNetDetector:
    def __init__(self, yunet_path: str, pad: float = 0.25, align: bool = True):
        if not os.path.isfile(yunet_path):
            raise FileNotFoundError(f"YuNet not found at {yunet_path}")
        self.det = cv.FaceDetectorYN_create(yunet_path, "", (320,240), 0.9, 0.3, 5000)
        self.pad = float(pad); self.align = bool(align)

    def _align_crop(self, crop: np.ndarray, le: Tuple[float,float], re: Tuple[float,float]) -> np.ndarray:
        dy, dx = re[1]-le[1], re[0]-le[0]
        angle = -math.degrees(math.atan2(dy, dx))  # negative to correct tilt
        ch, cw = crop.shape[:2]
        M = cv.getRotationMatrix2D((cw/2.0, ch/2.0), angle, 1.0)
        return cv.warpAffine(crop, M, (cw, ch), flags=cv.INTER_LINEAR)

    def crop112(self, bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]:
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces)==0: return None, None
        f = faces[0]
        x,y,ww,hh = map(float, f[:4])
        le=(float(f[4]),float(f[5])); re=(float(f[6]),float(f[7]))
        cx, cy = x+ww/2.0, y+hh/2.0
        side = max(ww, hh) * (1.0 + 2*self.pad)
        nx = int(max(0, cx - side/2.0)); ny = int(max(0, cy - side/2.0))
        nxe = int(min(w, cx + side/2.0)); nye = int(min(h, cy + side/2.0))
        crop = bgr[ny:nye, nx:nxe]
        if crop.size==0: return None, None
        if self.align:
            crop = self._align_crop(crop, (le[0]-nx, le[1]-ny), (re[0]-nx, re[1]-ny))
        face112 = cv.resize(crop, (112,112), interpolation=cv.INTER_AREA)
        return face112, (nx, ny, nxe-nx, nye-ny)

class FEROnnx:
    def __init__(self, onnx_path: str, labels_path: str):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"model.onnx not found: {onnx_path}")
        self.labels = load_labels(labels_path)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.oname = self.sess.get_outputs()[0].name

    def preprocess(self, bgr112: np.ndarray) -> np.ndarray:
        rgb = cv.cvtColor(bgr112, cv.COLOR_BGR2RGB).astype(np.float32)/255.0
        return np.transpose(rgb, (2,0,1))[None, ...]

    def predict(self, bgr112: np.ndarray, tta_flip: bool = True):
        x = self.preprocess(bgr112)
        logits = self.sess.run([self.oname], {self.iname: x})[0]
        if tta_flip:
            xflip = np.flip(x, axis=3).copy()  # mirror horizontally
            logits2 = self.sess.run([self.oname], {self.iname: xflip})[0]
            logits = (logits + logits2)/2.0
        probs = softmax(logits)
        idx = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        return self.labels[idx], conf, probs[0]

def annotate_and_save(img, box, text, out_path):
    vis = img.copy()
    if box:
        x,y,w,h = box
        cv.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.putText(vis, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    else:
        cv.putText(vis, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv.imwrite(str(out_path))

def process_image(path, fer, det, out_dir, save_vis):
    bgr = cv.imread(path)
    if bgr is None: log(f"READ FAIL: {path}"); return
    face, box = det.crop112(bgr) if det else (cv.resize(bgr,(112,112)), None)
    if face is None: log(f"NO FACE: {path}"); return
    label, conf, _ = fer.predict(face, tta_flip=True)
    log(f"IMAGE: {Path(path).name} -> {label} ({conf:.3f})")
    if save_vis:
        out_dir.mkdir(parents=True, exist_ok=True)
        annotate_and_save(bgr, box, f"{label} {conf:.2f}", out_dir/(Path(path).stem+"_pred.jpg"))

def process_folder(folder, fer, det, out_dir, save_vis):
    exts=(".jpg",".jpeg",".png",".bmp",".webp")
    paths=[str(p) for p in Path(folder).glob("**/*") if p.suffix.lower() in exts]
    for p in tqdm(paths, desc="Images"): process_image(p, fer, det, out_dir, save_vis)
    log(f"FOLDER DONE: {len(paths)} files")

def process_video(video_path, fer, det, out_dir, save_vis, display, write_fps):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened(): log(f"Cannot open video: {video_path}"); return
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer=None
    if save_vis:
        out_dir.mkdir(parents=True, exist_ok=True)
        outv = out_dir/(Path(video_path).stem+"_pred.mp4")
        writer = cv.VideoWriter(str(outv), cv.VideoWriter_fourcc(*"mp4v"), write_fps or fps, (W,H))
        log(f"Writing annotated video to {outv}")
    log(f"VIDEO OPEN: {video_path} ({W}x{H} @ {fps:.1f} fps) frames={total}")
    idx=0; detected=0; skipped=0; t0=time.time()
    pbar = tqdm(total=total if total>0 else None, desc="Frames")
    while True:
        ok, frame = cap.read()
        if not ok: break
        idx+=1
        face, box = det.crop112(frame) if det else (cv.resize(frame,(112,112)), None)
        if face is None:
            skipped+=1
            if writer: writer.write(frame)
            if display: 
                cv.imshow("FER", frame); 
                if (cv.waitKey(1)&0xFF)==27: break
            pbar.update(1); continue
        label, conf, _ = fer.predict(face, tta_flip=True)
        if box:
            x,y,w,h=box; 
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(frame,f"{label} {conf:.2f}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        else:
            cv.putText(frame,f"{label} {conf:.2f}",(20,40),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
        detected+=1
        if writer: writer.write(frame)
        if display:
            cv.imshow("FER", frame)
            if (cv.waitKey(1)&0xFF)==27: break
        pbar.update(1)
    cap.release(); 
    if writer: writer.release()
    if display: cv.destroyAllWindows()
    pbar.close()
    dt=time.time()-t0; eff_fps=(detected+skipped)/max(dt,1e-6)
    log(f"VIDEO DONE: frames={idx}, detected={detected}, skipped={skipped}, wall={dt:.2f}s, eff_fps={eff_fps:.2f}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--yunet", default="models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--no_align", action="store_true")
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--image", default=None)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--write_fps", type=float, default=None)
    args=ap.parse_args()

    fer = FEROnnx(args.onnx, args.labels)
    det = YuNetDetector(args.yunet, pad=args.pad, align=(not args.no_align)) if os.path.isfile(args.yunet) else None
    if det is None: log("WARN: YuNet missing/disabled; using whole-frame resize (less accurate).")
    out_dir = Path(args.out_dir)

    if args.image:   process_image(args.image, fer, det, out_dir, args.save_vis)
    elif args.folder:process_folder(args.folder, fer, det, out_dir, args.save_vis)
    elif args.video: process_video(args.video, fer, det, out_dir, args.save_vis, args.display, args.write_fps)
    else:
        log("Provide --image, --folder, or --video."); sys.exit(1)

if __name__=="__main__": main()












from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
import numpy as np
import cv2

app = FastAPI(title="Vision Emotion Service")

@app.post("/predict-vision")
async def predict_vision(file: UploadFile = File(...)):
    # 1. Read the uploaded image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")

    # 2. Run DeepFace emotion analysis
    #    actions can be any subset of ["age","gender","race","emotion"]
    result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)

    # 3. Extract what you need
    dominant_emotion = result["dominant_emotion"]
    emotion_scores   = result["emotion"]  # dict of all 7 emotion confidences

    return {
        "vision_emotion": dominant_emotion,
        "vision_scores": emotion_scores
    }
