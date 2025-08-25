import cv2
from deepface import DeepFace
import numpy as np

# Set up webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    try:
        # Analyze facial emotion with fast detector
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend='opencv',      # Faster than retinaface
            enforce_detection=False         # Skip errors if no face detected
        )

        # Extract dominant emotion
        dominant_emotion = result[0]['dominant_emotion']
        emotion_scores = result[0]['emotion']

        # Display dominant emotion on screen
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optionally show confidence scores for each emotion
        for idx, (emotion, score) in enumerate(emotion_scores.items()):
            text = f"{emotion}: {int(score)}%"
            cv2.putText(frame, text, (20, 80 + idx * 25),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

    except Exception as e:
        # In case of any issue (e.g. no face)
        cv2.putText(frame, "No face detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("⚠️ Error:", e)

    # Show the frame
    cv2.imshow("AI Kiosk - Real-Time Facial Emotion Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
