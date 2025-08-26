python run_fer.py --onnx model.onnx --labels labels.json --video demo.mp4 --yunet face_detection_yunet_2023mar.onnx --save_vis --display

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
