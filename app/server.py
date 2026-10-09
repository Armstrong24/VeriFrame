"""FastAPI backend: serves the dashboard and the /api/analyze endpoint.
Run:  uvicorn app.server:app --reload   then open http://127.0.0.1:8000
Model choice (first one found wins, or force one with the env var VERIFRAME_MODEL=v1plus|v2|v1):
  models/v1plus/  -> VeriFrame v1+ (v1 videos + v1 features, adds video-only mode)   [default]
  models/v2/      -> VeriFrame v2 (adds Whisper speech + CLAP audio, experimental)
  models/         -> original v1
"""
import sys, tempfile, os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_want = os.environ.get("VERIFRAME_MODEL", "").lower()
if _want in ("", "v1plus") and (ROOT / "models/v1plus/ensemble_v1plus.json").exists():
    from src.predictor_v1plus import PredictorV1Plus as _P
    VERSION = 3
elif _want in ("", "v2") and (ROOT / "models/v2/ensemble_v2.json").exists():
    from src.predictor_v2 import PredictorV2 as _P
    VERSION = 2
else:
    from src.predictor import Predictor as _P
    VERSION = 1

app = FastAPI(title="VeriFrame - Multimodal Fake Video Detector")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
predictor = _P()
print(f"[VeriFrame] loaded model version {VERSION}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/model-info")
def model_info():
    m = dict(predictor.metrics); m["version"] = VERSION
    return m


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), caption: str = Form(""), verified: int = Form(0), bio: str = Form("")):
    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read()); path = tmp.name
    try:
        import cv2                                   # reject files that are not readable videos
        cap_ = cv2.VideoCapture(path); ok_, _ = cap_.read(); cap_.release()
        if not ok_:
            raise HTTPException(400, "That file is not a readable video. Upload an .mp4 / .mov / .webm.")
        if VERSION >= 2:
            return predictor.analyze(path, caption, verified, bio)
        if not caption.strip():
            raise HTTPException(400, "v1 model needs a caption. Train v2 (scripts/05_train_v2.py) for video-only mode.")
        return predictor.analyze(path, caption, verified)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not analyze video: {e}")
    finally:
        os.unlink(path)
