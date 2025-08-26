# # import argparse, os, glob, json, numpy as np, cv2 as cv, onnxruntime as ort
# # from collections import Counter
# #
# # def load_labels(p):
# #     with open(p,'r',encoding='utf-8') as f: obj=json.load(f)
# #     if isinstance(obj,list): return obj
# #     if all(k.isdigit() for k in obj): return [obj[str(i)] for i in range(len(obj))]
# #     return [k for _,k in sorted(((int(v),k) for k,v in obj.items()))]
# #
# # def softmax(x):
# #     e=np.exp(x-x.max(axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)
# #
# # class YuNet:
# #     def __init__(self, onnx_path, pad=0.25, align=True):
# #         self.det=cv.FaceDetectorYN_create(onnx_path,"",(320,240),0.9,0.3,5000)
# #         self.pad=float(pad); self.align=bool(align)
# #     def crop112(self, bgr):
# #         h,w=bgr.shape[:2]; self.det.setInputSize((w,h))
# #         _,faces=self.det.detect(bgr)
# #         if faces is None or len(faces)==0: return None
# #         f=faces[0]
# #         x,y,ww,hh = [float(v) for v in f[:4]]
# #         cx,cy = x+ww/2.0, y+hh/2.0
# #         side  = max(ww,hh)*(1.0+2*self.pad)
# #         nx = int(max(0,cx-side/2.0)); ny = int(max(0,cy-side/2.0))
# #         nxe= int(min(w,cx+side/2.0)); nye= int(min(h,cy+side/2.0))
# #         crop=bgr[ny:nye,nx:nxe]
# #         if crop.size==0: return None
# #         if self.align:
# #             # eye landmarks are f[4:12]; quick alignment often helps but can be turned off
# #             le=(float(f[4])-nx, float(f[5])-ny); re=(float(f[6])-nx, float(f[7])-ny)
# #             dy,dx = re[1]-le[1], re[0]-le[0]
# #             angle = -np.degrees(np.arctan2(dy,dx))
# #             M=cv.getRotationMatrix2D((crop.shape[1]/2,crop.shape[0]/2),angle,1.0)
# #             crop=cv.warpAffine(crop,M,(crop.shape[1],crop.shape[0]),flags=cv.INTER_LINEAR)
# #         return cv.resize(crop,(112,112),interpolation=cv.INTER_AREA)
# #
# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--onnx", required=True)
# #     ap.add_argument("--labels", required=True)
# #     ap.add_argument("--root", required=True)      # test_set folder path
# #     ap.add_argument("--yunet", required=True)     # YuNet onnx path
# #     ap.add_argument("--pad", type=float, default=0.25)
# #     ap.add_argument("--align", action="store_true")
# #     ap.add_argument("--tta", action="store_true") # mirror TTA (slower, a bit more accurate)
# #     args = ap.parse_args()
# #
# #     labels = load_labels(args.labels); n=len(labels)
# #     lab2idx = {l:i for i,l in enumerate(labels)}
# #     det = YuNet(args.yunet, pad=args.pad, align=args.align)
# #
# #     # ONNXRuntime session
# #     opts = ort.SessionOptions(); opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# #     sess = ort.InferenceSession(args.onnx, sess_options=opts, providers=["CPUExecutionProvider"])
# #     iname = sess.get_inputs()[0].name; oname = sess.get_outputs()[0].name
# #
# #     def infer(face112):
# #         rgb=cv.cvtColor(face112,cv.COLOR_BGR2RGB).astype(np.float32)/255.0
# #         x=np.transpose(rgb,(2,0,1))[None,...]
# #         logits = sess.run([oname], {iname:x})[0]
# #         if args.tta:
# #             xflip = np.flip(x, axis=3).copy()
# #             logits = (logits + sess.run([oname], {iname:xflip})[0]) / 2.0
# #         p = softmax(logits); return int(np.argmax(p, axis=1)[0])
# #
# #     # collect files
# #     exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
# #     def files(cls):
# #         import itertools, os, glob
# #         p=os.path.join(args.root, cls)
# #         return list(itertools.chain.from_iterable(glob.glob(os.path.join(p,e)) for e in exts))
# #
# #     conf = np.zeros((n,n), dtype=int)
# #     totals=Counter(); correct=0; total=0
# #     noface=0
# #
# #     for lab in labels:
# #         for path in files(lab):
# #             img=cv.imread(path);
# #             if img is None: continue
# #             face = det.crop112(img)
# #             if face is None:
# #                 noface += 1
# #                 continue   # skip detection failures; report separately
# #             pred = infer(face)
# #             gt = lab2idx[lab]
# #             conf[gt, pred]+=1
# #             totals[lab]+=1
# #             correct += int(pred==gt); total += 1
# #
# #     overall = (correct/max(1,total))*100.0
# #     per_class = {labels[i]: (conf[i,i]/max(1,conf[i].sum())*100.0) for i in range(n)}
# #     macro = float(np.mean(list(per_class.values()))) if total>0 else 0.0
# #
# #     print(json.dumps({
# #         "overall_accuracy_percent": round(overall,2),
# #         "macro_accuracy_percent": round(macro,2),
# #         "samples_used": int(total),
# #         "skipped_no_face": int(noface),
# #         "per_class_percent": {k: round(v,2) for k,v in per_class.items()}
# #     }, indent=2))
# #     print("\nConfusion matrix (rows=true, cols=pred, order):", labels)
# #     print(conf)
# #
# # if __name__ == "__main__":
# #     import numpy as np
# # #     main()
# # # test_hse.py
# # import cv2 as cv, numpy as np, json
# # from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
# #
# # # init HSEmotion (downloads model once)
# # fer = HSEmotionRecognizer(model_name="enet_b2_7")  # 7-class AffectNet model
# #
# # # read one image (or grab from your webcam loop)
# # img = cv.imread("media/images/test1.jpg")
# # face = cv.resize(img, (112,112))  # or your YuNet crop112
# # emo, scores = fer.predict_emotions(face, logits=False)
# # print("Pred:", emo, "probs:", scores)
# import argparse, os, glob, json, numpy as np, cv2 as cv
# from collections import Counter
# from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
#
# def load_labels(path):
#     obj = json.load(open(path, "r", encoding="utf-8"))
#     if isinstance(obj, list): return obj
#     if all(k.isdigit() for k in obj): return [obj[str(i)] for i in range(len(obj))]
#     return [k for _,k in sorted(((int(v),k) for k,v in obj.items()))]
#
# def softmax(x):
#     e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_name", default="enet_b2_7")  # 7-class AffectNet
#     ap.add_argument("--labels", required=True)
#     ap.add_argument("--root", required=True)              # test_set/
#     ap.add_argument("--yunet", required=True)             # models/face_detection_yunet_2023mar.onnx
#     ap.add_argument("--pad", type=float, default=0.25)
#     ap.add_argument("--align", action="store_true")
#     args = ap.parse_args()
#
#     labels = load_labels(args.labels); n = len(labels); lab2idx = {l:i for i,l in enumerate(labels)}
#     # Face detector (YuNet)
#     det = cv.FaceDetectorYN_create(args.yunet, "", (320,240), 0.9, 0.3, 5000)
#
#     # HSEmotion (downloads/loads ONNX internally)
#     fer = HSEmotionRecognizer(model_name=args.model_name)
#
#     def crop112(img):
#         h,w = img.shape[:2]
#         det.setInputSize((w,h))
#         _, faces = det.detect(img)
#         if faces is None or len(faces)==0: return None
#         x,y,wf,hf = faces[0][:4].astype(int)
#         crop = img[max(0,y):y+hf, max(0,x):x+wf]
#         if crop.size==0: return None
#         return cv.resize(crop,(112,112), interpolation=cv.INTER_AREA)
#
#     exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
#     def files(cls):
#         import itertools
#         base = os.path.join(args.root, cls)
#         return list(itertools.chain.from_iterable(glob.glob(os.path.join(base,e)) for e in exts))
#
#     conf = np.zeros((n,n), dtype=int)
#     totals = Counter(); correct = 0; total = 0; noface = 0
#
#     for lab in labels:
#         for path in files(lab):
#             img = cv.imread(path)
#             if img is None: continue
#             face = crop112(img)
#             if face is None:
#                 noface += 1; continue
#             emo, probs = fer.predict_emotions(face, logits=False)
#             pred = lab2idx.get(emo.lower(), None)
#             if pred is None:
#                 # map capitalization/variants
#                 pred = lab2idx.get(emo.capitalize(), lab2idx.get(emo, None))
#             if pred is None:
#                 # fallback: find argmax by our label order
#                 p = np.array([probs.get(k,0.0) for k in labels])
#                 pred = int(np.argmax(p))
#             gt = lab2idx[lab]
#             conf[gt, pred] += 1
#             totals[lab] += 1
#             correct += int(pred == gt); total += 1
#
#     overall = (correct / max(1,total)) * 100.0
#     per_class = {labels[i]: (conf[i,i] / max(1, conf[i].sum()) * 100.0) for i in range(n)}
#     macro = float(np.mean(list(per_class.values()))) if total>0 else 0.0
#
#     print(json.dumps({
#         "model": args.model_name,
#         "overall_accuracy_percent": round(overall,2),
#         "macro_accuracy_percent": round(macro,2),
#         "samples_used": int(total),
#         "skipped_no_face": int(noface),
#         "per_class_percent": {k: round(v,2) for k,v in per_class.items()}
#     }, indent=2))
#     print("\nConfusion matrix (rows=true, cols=pred, order):", labels)
#     print(conf)
#
# if __name__ == "__main__":
#     import numpy as np, cv2 as cv
#     main()
#
#
#
import argparse, time, math, cv2 as cv, numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

LABELS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


class YuNet:
    def __init__(self, path, pad=0.25, align=False):
        self.det = cv.FaceDetectorYN_create(path, "", (320, 240), 0.9, 0.3, 5000)
        self.pad = float(pad);
        self.align = bool(align)

    def crop112(self, bgr):
        h, w = bgr.shape[:2];
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0: return None, None
        f = faces[0];
        x, y, ww, hh = [float(v) for v in f[:4]]
        cx, cy = x + ww / 2.0, y + hh / 2.0
        side = max(ww, hh) * (1.0 + 2 * self.pad)
        nx = int(max(0, cx - side / 2.0));
        ny = int(max(0, cy - side / 2.0))
        nxe = int(min(w, cx + side / 2.0));
        nye = int(min(h, cy + side / 2.0))
        crop = bgr[ny:nye, nx:nxe]
        if crop.size == 0: return None, None
        face = cv.resize(crop, (112, 112), interpolation=cv.INTER_AREA)
        return face, (nx, ny, nxe - nx, nye - ny)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_index", type=int, default=0)
    ap.add_argument("--backend", choices=["msmf", "dshow"], default="msmf")
    ap.add_argument("--yunet", default="models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--pad", type=float, default=0.30)
    ap.add_argument("--resize", default="640x480")
    args = ap.parse_args()

    W, H = [int(x) for x in args.resize.lower().split("x")]
    cap = cv.VideoCapture(args.cam_index, cv.CAP_MSMF if args.backend == "msmf" else cv.CAP_DSHOW)
    if not cap.isOpened(): raise RuntimeError("Camera open failed")
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, W);
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, H);
    cap.set(cv.CAP_PROP_FPS, 30)

    det = YuNet(args.yunet, pad=args.pad, align=False)  # start with align off (safer)
    fer = HSEmotionRecognizer(model_name="enet_b2_7")  # loads AffectNet 7-class

    t0 = time.time();
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames += 1
        face, box = det.crop112(frame)
        if face is not None:
            emo, probs = fer.predict_emotions(face, logits=False)  # emo is a string, e.g., "Fear"
            # probs is aligned to LABELS order:
            top = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)[:3]
            txt = f"{emo} {top[0][1]:.2f} | FPS {frames / (time.time() - t0 + 1e-6):.1f}"
            x, y, w, h = box
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, txt, (x, max(30, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.imshow("Live FER (ESC to quit)", frame)
        if (cv.waitKey(1) & 0xFF) == 27: break

    cap.release();
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()




