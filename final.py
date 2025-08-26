import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ------------- CONFIG -------------
# Path to the ResMaskNet weights you already have:
MODEL_PATH = "/Users/balreenkaur/Desktop/FaceEmotion/trained_models/ResMaskNet_Z_resmasking_dropout1_rot30.pth"

# Use OpenCV's built-in Haar face detector (fast & available offline)
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# FER-2013 style emotion labels commonly used with ResMaskNet
EMO_LABELS = ["angry", "disgust", "fearful", "happy", "sad", "surprised", "neutral"]

# Webcam index (0 is default). Use CAP_AVFOUNDATION for Mac stability
CAM_INDEX = 0
BACKEND = cv2.CAP_AVFOUNDATION
# ----------------------------------

# Import ResMaskNet wrapper from py-feat
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet


def get_torch_model(resmasknet_obj):
    """
    py-feat's ResMaskNet is a wrapper. Depending on version, the underlying
    torch model is exposed as .model or .net. We handle both.
    """
    inner = getattr(resmasknet_obj, "model", None) or getattr(resmasknet_obj, "net", None)
    if inner is None:
        raise RuntimeError("Could not access underlying torch model from ResMaskNet wrapper.")
    return inner


def preprocess_face(face_bgr, size=224):
    """
    Preprocess cropped face for ResNet-style models:
    - BGR -> RGB
    - resize to 224x224
    - convert to float tensor and normalize to ImageNet stats
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = face_resized.astype(np.float32) / 255.0

    # ImageNet mean/std (what ResNet expects)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    x = (x - mean) / std

    # HWC -> CHW -> batch
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return torch.from_numpy(x)


def draw_emotions(frame, box, probs, labels):
    """
    Draw dominant emotion + small score table near the face box.
    """
    x, y, w, h = box
    # Face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

    # Dominant
    idx = int(np.argmax(probs))
    dom = labels[idx]
    conf = int(round(probs[idx] * 100))
    cv2.putText(frame, f"{dom} ({conf}%)", (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    # Scores list
    start_y = y + h + 18
    for i, (lab, p) in enumerate(zip(labels, probs)):
        txt = f"{lab}: {int(round(p*100))}%"
        cv2.putText(frame, txt, (x, start_y + i*18),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def main():
    # ----- Load model -----
    try:
        # Many py-feat versions allow passing model_path
        model_wrap = ResMaskNet(model_path=MODEL_PATH)
    except TypeError:
        # Fallback if ctor signature doesn't accept model_path
        model_wrap = ResMaskNet()
        # If your package doesn't auto-load, patch the file path inside the package or re-install py-feat.

    torch_model = get_torch_model(model_wrap)
    torch_model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_model.to(device)

    # ----- Face detector -----
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade. Check HAAR_PATH.")

    # ----- Webcam -----
    cap = cv2.VideoCapture(CAM_INDEX, BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press 'q' to quit…")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                x_tensor = preprocess_face(face_roi)  # (1, 3, 224, 224)
                x_tensor = x_tensor.to(device)

                with torch.no_grad():
                    logits = torch_model(x_tensor)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                draw_emotions(frame, (x, y, w, h), probs, EMO_LABELS)

        cv2.imshow("AI Kiosk – ResNet (ResMaskNet) Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Sanity check: make sure the model file exists before running
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at:\n{MODEL_PATH}\nUpdate MODEL_PATH at the top of the script.")
    main()
