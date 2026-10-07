"""FastAPI backend: serves the dashboard and the /api/analyze endpoint.
Run:  uvicorn app.server:app --reload   then open http://127.0.0.1:8000
"""
import sys, tempfile, os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predictor import Predictor

app = FastAPI(title="VeriFrame - Multimodal Fake Video Detector")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
predictor = Predictor()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/model-info")
def model_info():
    return predictor.metrics


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), caption: str = Form(""), verified: int = Form(0)):
    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read()); path = tmp.name
    try:
        return predictor.analyze(path, caption, verified)
    except Exception as e:
        raise HTTPException(400, f"Could not analyze video: {e}")
    finally:
        os.unlink(path)
