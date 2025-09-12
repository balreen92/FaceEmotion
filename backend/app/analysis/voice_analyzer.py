from transformers import pipeline
import torch

MODEL_ID = "superb/hubert-large-superb-er"  # same as your script

# CPU or MPS on Mac; CUDA if available
def _device():
    if torch.cuda.is_available(): return 0
    return -1

_voice_pipe = None

def load_voice_model():
    global _voice_pipe
    if _voice_pipe is None:
        _voice_pipe = pipeline("audio-classification", model=MODEL_ID, device=_device())
    return _voice_pipe

def classify_voice(wav_path: str):
    clf = load_voice_model()
    results = clf(wav_path)
    if isinstance(results, dict):
        results = [results]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results[0]
    return {"label": top["label"], "score": float(top["score"]), "topk": results[:5]}
