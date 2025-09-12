import os, tempfile, math, cv2 as cv, ffmpeg

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def extract_audio_from_video(video_path: str, out_wav: str, sr=16000):
    ensure_dir(os.path.dirname(out_wav))
    (
        ffmpeg
        .input(video_path)
        .output(out_wav, ac=1, ar=sr, format='wav')
        .overwrite_output()
        .run(quiet=True)
    )
    return out_wav

def sample_video_frames(video_path: str, every_sec: float = 0.5, max_frames: int = 120):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    step = int(max(1, round(every_sec * fps)))
    idx, frames, count = 0, [], 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % step == 0:
            frames.append(frame.copy())
            count += 1
            if count >= max_frames: break
        idx += 1
    cap.release()
    return frames
