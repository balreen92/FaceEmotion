python run_fer.py --onnx model.onnx --labels labels.json --video demo.mp4 --yunet face_detection_yunet_2023mar.onnx --save_vis --display



# test_hse.py
import cv2 as cv, numpy as np, json
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# init HSEmotion (downloads model once)
fer = HSEmotionRecognizer(model_name="enet_b2_7")  # 7-class AffectNet model

# read one image (or grab from your webcam loop)
img = cv.imread("media\\images\\test.jpg")
face = cv.resize(img, (112,112))  # or your YuNet crop112
emo, scores = fer.predict_emotions(face, logits=False)
print("Pred:", emo, "probs:", scores)






import argparse, os, glob, json, numpy as np, cv2 as cv
from collections import Counter
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

def load_labels(path):
    obj = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(obj, list): return obj
    if all(k.isdigit() for k in obj): return [obj[str(i)] for i in range(len(obj))]
    return [k for _,k in sorted(((int(v),k) for k,v in obj.items()))]

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="enet_b2_7")  # 7-class AffectNet
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", required=True)              # test_set/
    ap.add_argument("--yunet", required=True)             # models/face_detection_yunet_2023mar.onnx
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--align", action="store_true")
    args = ap.parse_args()

    labels = load_labels(args.labels); n = len(labels); lab2idx = {l:i for i,l in enumerate(labels)}
    # Face detector (YuNet)
    det = cv.FaceDetectorYN_create(args.yunet, "", (320,240), 0.9, 0.3, 5000)

    # HSEmotion (downloads/loads ONNX internally)
    fer = HSEmotionRecognizer(model_name=args.model_name)

    def crop112(img):
        h,w = img.shape[:2]
        det.setInputSize((w,h))
        _, faces = det.detect(img)
        if faces is None or len(faces)==0: return None
        x,y,wf,hf = faces[0][:4].astype(int)
        crop = img[max(0,y):y+hf, max(0,x):x+wf]
        if crop.size==0: return None
        return cv.resize(crop,(112,112), interpolation=cv.INTER_AREA)

    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    def files(cls):
        import itertools
        base = os.path.join(args.root, cls)
        return list(itertools.chain.from_iterable(glob.glob(os.path.join(base,e)) for e in exts))

    conf = np.zeros((n,n), dtype=int)
    totals = Counter(); correct = 0; total = 0; noface = 0

    for lab in labels:
        for path in files(lab):
            img = cv.imread(path)
            if img is None: continue
            face = crop112(img)
            if face is None:
                noface += 1; continue
            emo, probs = fer.predict_emotions(face, logits=False)
            pred = lab2idx.get(emo.lower(), None)
            if pred is None:
                # map capitalization/variants
                pred = lab2idx.get(emo.capitalize(), lab2idx.get(emo, None))
            if pred is None:
                # fallback: find argmax by our label order
                p = np.array([probs.get(k,0.0) for k in labels])
                pred = int(np.argmax(p))
            gt = lab2idx[lab]
            conf[gt, pred] += 1
            totals[lab] += 1
            correct += int(pred == gt); total += 1

    overall = (correct / max(1,total)) * 100.0
    per_class = {labels[i]: (conf[i,i] / max(1, conf[i].sum()) * 100.0) for i in range(n)}
    macro = float(np.mean(list(per_class.values()))) if total>0 else 0.0

    print(json.dumps({
        "model": args.model_name,
        "overall_accuracy_percent": round(overall,2),
        "macro_accuracy_percent": round(macro,2),
        "samples_used": int(total),
        "skipped_no_face": int(noface),
        "per_class_percent": {k: round(v,2) for k,v in per_class.items()}
    }, indent=2))
    print("\nConfusion matrix (rows=true, cols=pred, order):", labels)
    print(conf)

if __name__ == "__main__":
    import numpy as np, cv2 as cv
    main()








import argparse, os, glob, json, numpy as np, cv2 as cv, onnxruntime as ort
from collections import Counter

def load_labels(p):
    with open(p,'r',encoding='utf-8') as f: obj=json.load(f)
    if isinstance(obj,list): return obj
    if all(k.isdigit() for k in obj): return [obj[str(i)] for i in range(len(obj))]
    return [k for _,k in sorted(((int(v),k) for k,v in obj.items()))]

def softmax(x):
    e=np.exp(x-x.max(axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)

class YuNet:
    def __init__(self, onnx_path, pad=0.25, align=True):
        self.det=cv.FaceDetectorYN_create(onnx_path,"",(320,240),0.9,0.3,5000)
        self.pad=float(pad); self.align=bool(align)
    def crop112(self, bgr):
        h,w=bgr.shape[:2]; self.det.setInputSize((w,h))
        _,faces=self.det.detect(bgr)
        if faces is None or len(faces)==0: return None
        f=faces[0]
        x,y,ww,hh = [float(v) for v in f[:4]]
        cx,cy = x+ww/2.0, y+hh/2.0
        side  = max(ww,hh)*(1.0+2*self.pad)
        nx = int(max(0,cx-side/2.0)); ny = int(max(0,cy-side/2.0))
        nxe= int(min(w,cx+side/2.0)); nye= int(min(h,cy+side/2.0))
        crop=bgr[ny:nye,nx:nxe]
        if crop.size==0: return None
        if self.align:
            # eye landmarks are f[4:12]; quick alignment often helps but can be turned off
            le=(float(f[4])-nx, float(f[5])-ny); re=(float(f[6])-nx, float(f[7])-ny)
            dy,dx = re[1]-le[1], re[0]-le[0]
            angle = -np.degrees(np.arctan2(dy,dx))
            M=cv.getRotationMatrix2D((crop.shape[1]/2,crop.shape[0]/2),angle,1.0)
            crop=cv.warpAffine(crop,M,(crop.shape[1],crop.shape[0]),flags=cv.INTER_LINEAR)
        return cv.resize(crop,(112,112),interpolation=cv.INTER_AREA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", required=True)      # test_set folder path
    ap.add_argument("--yunet", required=True)     # YuNet onnx path
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--tta", action="store_true") # mirror TTA (slower, a bit more accurate)
    args = ap.parse_args()

    labels = load_labels(args.labels); n=len(labels)
    lab2idx = {l:i for i,l in enumerate(labels)}
    det = YuNet(args.yunet, pad=args.pad, align=args.align)

    # ONNXRuntime session
    opts = ort.SessionOptions(); opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, sess_options=opts, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name; oname = sess.get_outputs()[0].name

    def infer(face112):
        rgb=cv.cvtColor(face112,cv.COLOR_BGR2RGB).astype(np.float32)/255.0
        x=np.transpose(rgb,(2,0,1))[None,...]
        logits = sess.run([oname], {iname:x})[0]
        if args.tta:
            xflip = np.flip(x, axis=3).copy()
            logits = (logits + sess.run([oname], {iname:xflip})[0]) / 2.0
        p = softmax(logits); return int(np.argmax(p, axis=1)[0])

    # collect files
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    def files(cls):
        import itertools, os, glob
        p=os.path.join(args.root, cls)
        return list(itertools.chain.from_iterable(glob.glob(os.path.join(p,e)) for e in exts))

    conf = np.zeros((n,n), dtype=int)
    totals=Counter(); correct=0; total=0
    noface=0

    for lab in labels:
        for path in files(lab):
            img=cv.imread(path); 
            if img is None: continue
            face = det.crop112(img)
            if face is None:
                noface += 1
                continue   # skip detection failures; report separately
            pred = infer(face)
            gt = lab2idx[lab]
            conf[gt, pred]+=1
            totals[lab]+=1
            correct += int(pred==gt); total += 1

    overall = (correct/max(1,total))*100.0
    per_class = {labels[i]: (conf[i,i]/max(1,conf[i].sum())*100.0) for i in range(n)}
    macro = float(np.mean(list(per_class.values()))) if total>0 else 0.0

    print(json.dumps({
        "overall_accuracy_percent": round(overall,2),
        "macro_accuracy_percent": round(macro,2),
        "samples_used": int(total),
        "skipped_no_face": int(noface),
        "per_class_percent": {k: round(v,2) for k,v in per_class.items()}
    }, indent=2))
    print("\nConfusion matrix (rows=true, cols=pred, order):", labels)
    print(conf)

if __name__ == "__main__":
    import numpy as np
    main()









import cv2 as cv, time

def try_open(desc, cap):
    ok = cap.isOpened()
    print(f"[{desc}] opened:", ok)
    if not ok: 
        return False
    ok2, frame = cap.read()
    print(f"[{desc}] read frame:", ok2, ("shape="+str(frame.shape) if ok2 else ""))
    if ok2:
        cv.imshow(desc, frame)
        cv.waitKey(500)  # brief preview
        cv.destroyAllWindows()
    cap.release()
    return ok and ok2

print("Backends: DSHOW (DirectShow), MSMF (Media Foundation). Trying indices 0..4 and Camo by name.\n")

# 1) Try numeric indices with DirectShow & MSMF
for backend, name in [(cv.CAP_DSHOW,"DSHOW"), (cv.CAP_MSMF,"MSMF")]:
    for i in range(5):
        cap = cv.VideoCapture(i, backend)
        try_open(f"{name} index {i}", cap)

# 2) Try opening by device name via DirectShow (works on many systems)
# Replace the string below if your device name differs in Device Manager / Camo Studio
for devname in ["video=Reincubate Camo", "video=Camo", "video=EpocCam Camera", "video=DroidCam Source 3"]:
    cap = cv.VideoCapture(devname, cv.CAP_DSHOW)
    try_open(f"DSHOW name '{devname}'", cap)

print("\nDone.")







import cv2 as cv
for i in range(5):
    cap = cv.VideoCapture(i, cv.CAP_DSHOW)
    ok = cap.isOpened()
    print(f"Index {i}: {'OPEN' if ok else 'FAIL'}")
    if ok:
        ret, frame = cap.read()
        print("  Frame:", "OK" if ret else "NO")
        cap.release()


import argparse, time, math, json, numpy as np, cv2 as cv, onnxruntime as ort

def load_labels(p):
    obj=json.load(open(p,"r"))
    if isinstance(obj,list): return obj
    if all(k.isdigit() for k in obj): return [obj[str(i)] for i in range(len(obj))]
    return [k for _,k in sorted(((int(v),k) for k,v in obj.items()))]

def softmax(x): e=np.exp(x-x.max(axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)

class YuNet:
    def __init__(self, path, pad=0.25, align=True):
        self.det=cv.FaceDetectorYN_create(path,"",(320,240),0.9,0.3,5000)
        self.pad=float(pad); self.align=align
    def crop112(self, bgr):
        h,w=bgr.shape[:2]; self.det.setInputSize((w,h))
        _,faces=self.det.detect(bgr)
        if faces is None or len(faces)==0: return None,None
        f=faces[0]; x,y,ww,hh=[float(v) for v in f[:4]]
        le=(float(f[4]),float(f[5])); re=(float(f[6]),float(f[7]))
        cx,cy=x+ww/2,y+hh/2; side=max(ww,hh)*(1+2*self.pad)
        nx=int(max(0,cx-side/2)); ny=int(max(0,cy-side/2))
        nxe=int(min(w,cx+side/2)); nye=int(min(h,cy+side/2))
        crop=bgr[ny:nye,nx:nxe]
        if crop.size==0: return None,None
        if self.align:
            dy,dx=re[1]-le[1],re[0]-le[0]
            angle=-math.degrees(math.atan2(dy,dx))
            M=cv.getRotationMatrix2D((crop.shape[1]/2,crop.shape[0]/2),angle,1.0)
            crop=cv.warpAffine(crop,M,(crop.shape[1],crop.shape[0]),flags=cv.INTER_LINEAR)
        face=cv.resize(crop,(112,112),interpolation=cv.INTER_AREA)
        return face,(nx,ny,nxe-nx,nye-ny)

class FER:
    def __init__(self, onnx, labels, tta=False, coreml=False):
        self.labels=load_labels(labels)
        opts=ort.SessionOptions(); opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers=["CPUExecutionProvider"]
        if coreml: providers=["CoreMLExecutionProvider","CPUExecutionProvider"]
        self.sess=ort.InferenceSession(onnx, sess_options=opts, providers=providers)
        self.inp=self.sess.get_inputs()[0].name; self.out=self.sess.get_outputs()[0].name
        self.tta=tta
    def predict(self, face_bgr_112):
        rgb=cv.cvtColor(face_bgr_112,cv.COLOR_BGR2RGB).astype(np.float32)/255.0
        x=np.transpose(rgb,(2,0,1))[None,...]
        logits=self.sess.run([self.out],{self.inp:x})[0]
        if self.tta:
            xflip=np.flip(x,axis=3).copy()
            logits=(logits+self.sess.run([self.out],{self.inp:xflip})[0])/2.0
        p=softmax(logits); idx=int(np.argmax(p,axis=1)[0]); conf=float(np.max(p,axis=1)[0])
        return self.labels[idx], conf

def open_cam(index=None, url=None):
    if url:
        print(f"[INFO] Opening IP cam: {url}", flush=True)
        cap=cv.VideoCapture(url)
    else:
        i=0 if index is None else index
        print(f"[INFO] Opening device index: {i}", flush=True)
        cap=cv.VideoCapture(i, cv.CAP_DSHOW)  # stable on Windows
    if not cap.isOpened(): raise RuntimeError("Could not open camera/stream")
    return cap

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/model.onnx")
    ap.add_argument("--labels", default="models/labels.json")
    ap.add_argument("--yunet", default="models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--cam_index", type=int, default=0)
    ap.add_argument("--ipcam_url", type=str, default=None)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--resize", type=str, default="640x480")  # downscale for speed
    ap.add_argument("--tta", action="store_true")             # mirror TTA (slower)
    ap.add_argument("--show_fps", action="store_true")
    args=ap.parse_args()

    W,H=map(int,args.resize.lower().split("x"))
    det=YuNet(args.yunet, pad=args.pad, align=args.align)
    fer=FER(args.onnx, args.labels, tta=args.tta)

    cap=open_cam(args.cam_index, args.ipcam_url)
    t0=time.time(); frames=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv.resize(frame,(W,H))
        face,box=det.crop112(frame)
        if face is not None:
            label,conf=fer.predict(face)
            x,y,w,h=box
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            txt=f"{label} {conf:.2f}"
            if args.show_fps:
                frames+=1; fps=frames/(time.time()-t0+1e-6)
                txt+=f" | {fps:.1f} FPS"
            cv.putText(frame,txt,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv.imshow("Live FER (ESC to quit)", frame)
        if (cv.waitKey(1)&0xFF)==27: break
    cap.release(); cv.destroyAllWindows()

if __name__=="__main__": main()











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
