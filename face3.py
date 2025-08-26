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
        cap = cv.VideoCapture(0, cv.CAP_MSMF)   # stable on Windows
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
