# multimodal_emotion_bot.py

import cv2
import pyttsx3
import speech_recognition as sr
from fer import FER
from textblob import TextBlob


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."


def detect_facial_emotion():
    detector = FER(mtcnn=True)
    cam = cv2.VideoCapture(0)
    print("Opening camera. Press 'q' to exit.")

    emotion_result = ""
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        emotions = detector.detect_emotions(frame)
        if emotions:
            top_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            emotion_result = top_emotion
            print(f"Detected face emotion: {top_emotion}")

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return emotion_result


def detect_text_emotion(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "joy"
    elif polarity < -0.3:
        return "sad"
    else:
        return "neutral"


def main():
    speak("Hello! Let's talk. How are you feeling today?")

    # Voice-based emotion (via text sentiment)
    user_text = listen()
    print(f"User said: {user_text}")
    text_emotion = detect_text_emotion(user_text)

    # Face-based emotion
    facial_emotion = detect_facial_emotion()

    print(f"Text Emotion: {text_emotion}, Facial Emotion: {facial_emotion}")

    final_emotion = facial_emotion if facial_emotion else text_emotion
    response = f"I sense that you're feeling {final_emotion}."
    speak(response)


if __name__ == "__main__":
    main()
