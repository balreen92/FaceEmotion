
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'feat', 'feat')))


# Now this will work
from emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
from torchvision import transforms

# --- Load face detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Define device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load model ---
model = ResMaskNet(device=device, pretrained="local")
model.model.load_state_dict(
    torch.load("trained_models/ResMaskNet_Z_resmasking_dropout1_rot30.pth", map_location=device)["net"]
)
model.model.eval()


# --- Preprocessing for face input ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Emotion Labels (edit if your model uses different ones) ---
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Start Webcam ---
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  #  Mirror horizontally

    frame = cv2.resize(frame, (640, 480))  #  Resize to 640x480
    frame_count += 1
    if frame_count % 2 != 0:  # skip every other frame
        continue

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w]
        face_tensor = transform(cv2.resize(face_crop, (224, 224))).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.model(face_tensor)

            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            label = emotions[pred]
            confidence = torch.max(prob).item() * 100

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
