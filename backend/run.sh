#!/usr/bin/env bash
export YUNET_PATH="models/face_detection_yunet_2023mar.onnx"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
