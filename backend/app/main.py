import os, tempfile, uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.utils.media import extract_audio_from_video, sample_video_frames
from app.analysis.voice_analyzer import classify_voice
from app.analysis.face_analyzer import classify_faces_from_frames

YUNET_PATH = os.getenv("YUNET_PATH", "models/face_detection_yunet_2023mar.onnx")

app = FastAPI(title="Emotion Kiosk API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class FusionOut(BaseModel):
    face: dict
    voice: dict
    fused_label: str
    fused_score: float

@app.get("/health")
def health(): return {"ok": True}

@app.post("/api/upload-video", response_model=FusionOut)
async def upload_video(video: UploadFile = File(...), session_id: str = Form("na")):
    # Save upload
    tmpdir = tempfile.mkdtemp(prefix="kiosk_")
    vid_path = os.path.join(tmpdir, f"{uuid.uuid4()}.webm")
    with open(vid_path, "wb") as f:
        f.write(await video.read())

    # 1) Voice
    wav_path = os.path.join(tmpdir, "audio.wav")
    extract_audio_from_video(vid_path, wav_path, sr=16000)
    voice_result = classify_voice(wav_path)

    # 2) Face (average across frames)
    frames = sample_video_frames(vid_path, every_sec=0.5, max_frames=180)
    face_result = classify_faces_from_frames(frames, YUNET_PATH)

    # 3) Simple fusion (weights: face 0.6, voice 0.4)
    weights = {"face": 0.6, "voice": 0.4}
    fused_map = {}
    def add_score(d, w):
        fused_map[d["label"]] = fused_map.get(d["label"], 0.0) + d["score"] * w
    add_score(face_result, weights["face"])
    add_score(voice_result, weights["voice"])
    fused_label = max(fused_map, key=fused_map.get)
    fused_score = float(fused_map[fused_label])

    return {
        "face": face_result,
        "voice": voice_result,
        "fused_label": fused_label,
        "fused_score": fused_score
    }
