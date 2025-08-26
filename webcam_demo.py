import cv2 as cv, numpy as np

FER_ONNX = "facial_expression_recognition_mobilefacenet_2022july.onnx"
YUNET_ONNX = "face_detection_yunet_2023mar.onnx"
LABELS = [l.strip() for l in open("labels.txt")]

fer = cv.dnn.readNet(FER_ONNX)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Grab a frame to know width/height for YuNet init
ok, frame = cap.read()
if not ok:
    raise RuntimeError("No frames from webcam")
h, w = frame.shape[:2]

# Create YuNet detector (this is the correct API)
det = cv.FaceDetectorYN_create(
    YUNET_ONNX,
    "",                 # no config file
    (w, h),             # input size must be set and kept updated
    0.9,                # score threshold
    0.3,                # NMS threshold
    5000                # topK
)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]

    # YuNet requires you to set the input size whenever frame size changes
    det.setInputSize((w, h))

    # detect returns (num_faces, faces); faces: Nx15 [x,y,w,h,5*landmarks,score]
    _, faces = det.detect(frame)
    if faces is not None:
        for f in faces:
            x, y, ww, hh = f[:4].astype(int)
            x, y = max(x, 0), max(y, 0)
            crop = frame[y:y+hh, x:x+ww]
            if crop.size == 0:
                continue

            blob = cv.dnn.blobFromImage(
                crop, scalefactor=1/255.0, size=(112, 112),
                mean=(0,0,0), swapRB=True, crop=True
            )
            fer.setInput(blob)
            prob = fer.forward().squeeze()
            idx = int(np.argmax(prob))
            conf = float(prob[idx])

            cv.rectangle(frame, (x, y), (x+ww, y+hh), (0,255,0), 2)
            cv.putText(frame, f"{LABELS[idx]} {conf:.2f}", (x, y-8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv.imshow("FER demo", frame)
    if cv.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
