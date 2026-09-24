import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .detector import PPEDetector
from .video_stream import ThreadedVideoStream

app = FastAPI(title="PPE Safety Detection Pipeline")
detector = PPEDetector()
_stream: ThreadedVideoStream = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "invalid image"})

    detections = detector.detect(frame)
    missing = detector.missing_ppe(detections)

    return {
        "detections": [d.__dict__ for d in detections],
        "missing_ppe": sorted(missing),
        "compliant": len(missing) == 0,
    }


@app.on_event("startup")
def start_stream():
    global _stream
    try:
        _stream = ThreadedVideoStream(0).start()
    except RuntimeError:
        _stream = None


@app.on_event("shutdown")
def stop_stream():
    if _stream is not None:
        _stream.stop()


@app.get("/stream/violations")
def stream_violations():
    if _stream is None:
        return JSONResponse(status_code=503, content={"error": "no camera stream available"})

    frame = _stream.read()
    if frame is None:
        return JSONResponse(status_code=503, content={"error": "no frame available yet"})

    detections = detector.detect(frame)
    missing = detector.missing_ppe(detections)
    return {"missing_ppe": sorted(missing), "compliant": len(missing) == 0}
