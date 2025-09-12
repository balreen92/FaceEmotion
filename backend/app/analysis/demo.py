# demo_realtime.py
import cv2, time
from face_accelerated import FaceEmotionAnalyzer, YuNet, YUNET_PATH

an = FaceEmotionAnalyzer()
det_vis = YuNet(YUNET_PATH, pad=0.20, score=0.6)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # ← your working camera index
start = time.time(); duration = 20

while True:
    ret, frame = cap.read()
    if not ret: break

    # visualize detection box
    faces = det_vis.detect(frame)
    if faces is not None and len(faces)>0:
        x,y,w,h = [int(v) for v in faces[0][:4]]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    out = an.classify_frames([frame])
    cv2.putText(frame, f"{out['label']} ({out['score']:.2f})", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("FER (Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if time.time() - start > duration: break

cap.release(); cv2.destroyAllWindows()
