
#!/usr/bin/env python3

"""
Quick ONNX inference: single image or webcam.
Usage:
  # Image
  python fer_infer_onnx.py --onnx outputs/model.onnx --labels outputs/labels.json --image /path/img.jpg --yunet face_detection_yunet_2023mar.onnx --align

  # Webcam
  python fer_infer_onnx.py --onnx outputs/model.onnx --labels outputs/labels.json --webcam 0 --yunet face_detection_yunet_2023mar.onnx --align
"""
import argparse, json, time, numpy as np, cv2 as cv, onnxruntime as ort

def load_labels(path):
    with open(path,"r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and all(k.isdigit() for k in obj.keys()):
        return [obj[str(i)] for i in range(len(obj))]
    elif isinstance(obj, dict):
        items = sorted(((int(v),k) for k,v in obj.items()), key=lambda x:x[0])
        return [k for _,k in items]
    else:
        return list(obj)

class YuNet:
    def __init__(self, onnx_path, pad=0.25, align=False):
        self.pad=pad; self.align=align
        self.det = cv.FaceDetectorYN_create(onnx_path, "", (320,240), 0.9, 0.3, 5000)
    def crop112(self, img_bgr):
        h,w = img_bgr.shape[:2]
        self.det.setInputSize((w,h))
        _, faces = self.det.detect(img_bgr)
        if faces is None or len(faces)==0: return None, None
        f = faces[0]
        x,y,ww,hh = [float(v) for v in f[:4]]
        le=(float(f[4]),float(f[5])); re=(float(f[6]),float(f[7]))
        cx,cy = x+ww/2.0, y+hh/2.0
        side = max(ww,hh)*(1.0+2*self.pad)
        nx=int(max(0,cx-side/2.0)); ny=int(max(0,cy-side/2.0))
        nxe=int(min(w,cx+side/2.0)); nye=int(min(h,cy+side/2.0))
        crop = img_bgr[ny:nye, nx:nxe]
        if crop.size==0: return None, None
        if self.align:
            dy=re[1]-le[1]; dx=re[0]-le[0]
            angle = -np.degrees(np.arctan2(dy, dx))
            ch,cw = crop.shape[:2]
            M = cv.getRotationMatrix2D((cw/2.0, ch/2.0), angle, 1.0)
            crop = cv.warpAffine(crop, M, (cw,ch), flags=cv.INTER_LINEAR)
        face112 = cv.resize(crop, (112,112), interpolation=cv.INTER_AREA)
        return face112, (int(nx),int(ny),int(nxe-nx),int(nye-ny))

def preprocess(bgr):
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = np.transpose(rgb,(2,0,1))[None,...]
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--webcam", type=int, default=None)
    ap.add_argument("--yunet", default=None)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--align", action="store_true")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    cropper = YuNet(args.yunet, pad=args.pad, align=args.align) if args.yunet else None

    def predict(face112):
        x = preprocess(face112)
        logits = sess.run([out_name], {in_name:x})[0]
        idx = int(np.argmax(logits, axis=1)[0])
        conf = float(np.max(softmax(logits), axis=1)[0])
        return idx, conf

    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

    if args.image:
        img = cv.imread(args.image)
        if img is None:
            print("Failed to read image"); return
        if cropper:
            face, box = cropper.crop112(img)
            if face is None:
                print("No face"); return
        else:
            face = cv.resize(img, (112,112), interpolation=cv.INTER_AREA); box=None
        idx, conf = predict(face)
        if box:
            x,y,w,h = box
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(img, f"{labels[idx]} {conf:.2f}", (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv.imshow("result", img); cv.waitKey(0); cv.destroyAllWindows()
        print({"label": labels[idx], "confidence": round(conf,3)})
        return

    if args.webcam is not None:
        cap = cv.VideoCapture(args.webcam)
        if not cap.isOpened():
            print("Cannot open webcam"); return
        while True:
            ok, frame = cap.read()
            if not ok: break
            if cropper:
                face, box = cropper.crop112(frame)
                if face is not None:
                    idx, conf = predict(face)
                    if box:
                        x,y,w,h = box
                        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        cv.putText(frame, f"{labels[idx]} {conf:.2f}", (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv.imshow("FER ONNX", frame)
            if (cv.waitKey(1)&0xFF)==27: break
        cap.release(); cv.destroyAllWindows()

if __name__ == "__main__":
    main()
