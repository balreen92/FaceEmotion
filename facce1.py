import argparse, json, os, time, math, sys
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2 as cv
import onnxruntime as ort
from tqdm import tqdm


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # allow ["angry", ...] or {"0":"angry", ...} or {"angry":0, ...}
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if all(k.isdigit() for k in obj.keys()):  # {"0":"angry",...}
            return [obj[str(i)] for i in range(len(obj))]
        else:  # {"angry":0,...} -> sort by index
            inv = sorted(((int(v), k) for k, v in obj.items()), key=lambda x: x[0])
            return [k for _, k in inv]
    raise ValueError("labels.json format unsupported")

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# --------------------------- YuNet Detector ---------------------------

class YuNetDetector:
    """
    OpenCV Zoo YuNet face detector with optional eye alignment and padding.
    """
    def __init__(self, yunet_path: str, pad: float = 0.25, align: bool = True):
        if not os.path.isfile(yunet_path):
            raise FileNotFoundError(f"YuNet not found at {yunet_path}")
        self.det = cv.FaceDetectorYN_create(yunet_path, "", (320, 240), 0.9, 0.3, 5000)
        self.pad = float(pad)
        self.align = bool(align)

    def _align_crop(self, crop: np.ndarray, left_eye: Tuple[float,float], right_eye: Tuple[float,float]) -> np.ndarray:
        # rotate to make eyes horizontal (negative angle to correct tilt)
        dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = -math.degrees(math.atan2(dy, dx))
        ch, cw = crop.shape[:2]
        M = cv.getRotationMatrix2D((cw/2.0, ch/2.0), angle, 1.0)
        return cv.warpAffine(crop, M, (cw, ch), flags=cv.INTER_LINEAR)

    def crop112(self, bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]:
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None, None

        f = faces[0]
        x, y, ww, hh = map(float, f[:4])
        le = (float(f[4]),  float(f[5]))   # left eye
        re = (float(f[6]),  float(f[7]))   # right eye

        cx, cy = x + ww/2.0, y + hh/2.0
        side = max(ww, hh) * (1.0 + 2*self.pad)
        nx = int(max(0, cx - side/2.0)); ny = int(max(0, cy - side/2.0))
        nxe = int(min(w, cx + side/2.0)); nye = int(min(h, cy + side/2.0))
        crop = bgr[ny:nye, nx:nxe]
        if crop.size == 0:
            return None, None

        if self.align:
            crop = self._align_crop(crop, (le[0]-nx, le[1]-ny), (re[0]-nx, re[1]-ny))

        face112 = cv.resize(crop, (112, 112), interpolation=cv.INTER_AREA)
        return face112, (nx, ny, nxe - nx, nye - ny)

# --------------------------- ONNX Inference ---------------------------

class FEROnnx:
    def __init__(self, onnx_path: str, labels_path: str, providers=None):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"model.onnx not found at {onnx_path}")
        self.labels = load_labels(labels_path)
        prov = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=prov)
        self.iname = self.sess.get_inputs()[0].name
        self.oname = self.sess.get_outputs()[0].name

    def preprocess(self, bgr112: np.ndarray) -> np.ndarray:
        rgb = cv.cvtColor(bgr112, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]  # 1x3x112x112
        return x

    def predict(self, bgr112: np.ndarray, tta_flip: bool = True) -> Tuple[str, float, np.ndarray]:
        x = self.preprocess(bgr112)
        logits = self.sess.run([self.oname], {self.iname: x})[0]
        if tta_flip:
            xflip = np.flip(x, axis=3).copy()
            logits2 = self.sess.run([self.oname], {self.iname: xflip})[0]
            logits = (logits + logits2) / 2.0
        probs = softmax(logits)
        idx = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        return self.labels[idx], conf, probs[0]

# ------------------------------- Runner -------------------------------

def process_image(path: str, fer: FEROnnx, det: Optional[YuNetDetector], save_vis: bool, out_dir: Path):
    bgr = cv.imread(path)
    if bgr is None:
        log(f"READ FAIL: {path}")
        return False

    face, box = (None, None)
    if det:
        face, box = det.crop112(bgr)
        if face is None:
            log(f"NO FACE: {path}")
            return False
    else:
        face = cv.resize(bgr, (112,112), interpolation=cv.INTER_AREA)

    label, conf, _ = fer.predict(face, tta_flip=True)
    log(f"IMAGE: {Path(path).name} -> {label} ({conf:.3f})")

    if save_vis:
        vis = bgr.copy()
        if box:
            x,y,w,h = box
            cv.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.putText(vis, f"{label} {conf:.2f}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv.putText(vis, f"{label} {conf:.2f}", (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        out_dir.mkdir(parents=True, exist_ok=True)
        outp = out_dir / (Path(path).stem + "_pred.jpg")
        cv.imwrite(str(outp), vis)
    return True

def process_folder(folder: str, fer: FEROnnx, det: Optional[YuNetDetector], save_vis: bool, out_dir: Path):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    paths = [str(p) for p in Path(folder).glob("**/*") if p.suffix.lower() in exts]
    if not paths:
        log("No images found in folder.")
        return
    ok, miss = 0, 0
    for p in tqdm(paths, desc="Images"):
        if process_image(p, fer, det, save_vis, out_dir):
            ok += 1
        else:
            miss += 1
    log(f"FOLDER DONE: {ok} ok, {miss} skipped/fail")

def process_video(video_path: str, fer: FEROnnx, det: Optional[YuNetDetector],
                  save_vis: bool, out_dir: Path, display: bool, write_fps: Optional[float]):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Cannot open video: {video_path}")
        return
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_vis:
        out_dir.mkdir(parents=True, exist_ok=True)
        outv = out_dir / (Path(video_path).stem + "_pred.mp4")
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(str(outv), fourcc, write_fps or fps, (W, H))
        log(f"Writing annotated video to {outv} at {write_fps or fps:.1f} fps")

    log(f"VIDEO OPEN: {video_path} ({W}x{H} @ {fps:.1f} fps) frames={total}")
    frame_idx, detected, skipped = 0, 0, 0
    pbar = tqdm(total=total if total>0 else None, desc="Frames")
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        face, box = (None, None)
        if det:
            face, box = det.crop112(frame)
        else:
            face = cv.resize(frame, (112,112), interpolation=cv.INTER_AREA)

        if face is None:
            skipped += 1
            if writer: writer.write(frame)
            if display:
                cv.imshow("FER", frame)
                if (cv.waitKey(1) & 0xFF) == 27: break
            pbar.update(1)
            continue

        label, conf, _ = fer.predict(face, tta_flip=True)
        if box:
            x,y,w,h = box
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv.putText(frame, f"{label} {conf:.2f}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv.putText(frame, f"{label} {conf:.2f}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        detected += 1
        if writer: writer.write(frame)
        if display:
            cv.imshow("FER", frame)
            if (cv.waitKey(1) & 0xFF) == 27:
                break
        pbar.update(1)

    t_total = time.time() - t_start
    cap.release()
    if writer: writer.release()
    if display: cv.destroyAllWindows()
    pbar.close()
    fps_eff = (detected + skipped) / max(1e-6, t_total)
    log(f"VIDEO DONE: frames={frame_idx}, detected={detected}, skipped={skipped}, wall={t_total:.2f}s, eff_fps={fps_eff:.2f}")

# ------------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to model.onnx")
    ap.add_argument("--labels", required=True, help="Path to labels.json (list or dict)")
    ap.add_argument("--yunet", default="face_detection_yunet_2023mar.onnx", help="YuNet ONNX")
    ap.add_argument("--no_align", action="store_true", help="Disable eye alignment")
    ap.add_argument("--pad", type=float, default=0.25, help="Padding ratio around face bbox")
    ap.add_argument("--image", default=None, help="Run on a single image")
    ap.add_argument("--folder", default=None, help="Run on a folder of images (recursive)")
    ap.add_argument("--video",  default=None, help="Run on a video file")
    ap.add_argument("--save_vis", action="store_true", help="Save annotated outputs")
    ap.add_argument("--out_dir", default="outputs", help="Output directory")
    ap.add_argument("--display", action="store_true", help="Show a window while processing")
    ap.add_argument("--write_fps", type=float, default=None, help="Override output video FPS")
    args = ap.parse_args()

    labels = load_labels(args.labels)  # sanity check
    log(f"Labels: {labels}")

    fer = FEROnnx(args.onnx, args.labels)
    det = YuNetDetector(args.yunet, pad=args.pad, align=(not args.no_align)) if os.path.isfile(args.yunet) else None
    if det is None:
        log("WARN: YuNet not found or disabled; will resize whole frame to 112x112 (lower accuracy).")

    out_dir = Path(args.out_dir)

    if args.image:
        log("MODE: single image")
        process_image(args.image, fer, det, args.save_vis, out_dir)
    elif args.folder:
        log("MODE: folder")
        process_folder(args.folder, fer, det, args.save_vis, out_dir)
    elif args.video:
        log("MODE: video")
        process_video(args.video, fer, det, args.save_vis, out_dir, display=args.display, write_fps=args.write_fps)
    else:
        log("Nothing to do. Provide --image, --folder, or --video.")
        sys.exit(1)

if __name__ == "__main__":
    main()