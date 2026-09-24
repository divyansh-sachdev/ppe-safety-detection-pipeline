<div align="center">

# PPE Safety Detection Pipeline

**Real-time PPE compliance detection with YOLOv11 and CUDA-accelerated multi-threaded inference**

![Domain](https://img.shields.io/badge/Domain-Computer_Vision_/_Deep_Learning-00F3FF?style=for-the-badge) ![Model](https://img.shields.io/badge/Model-YOLOv11-9D00FF?style=for-the-badge) ![Precision](https://img.shields.io/badge/Precision-96.5%_mAP@50-0066FF?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-0D1117?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-0D1117?style=flat-square&logo=pytorch&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-0D1117?style=flat-square&logo=opencv&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-0D1117?style=flat-square&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-0D1117?style=flat-square&logo=docker&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-0D1117?style=flat-square&logo=nvidia&logoColor=white)

</div>

---

## Overview

A real-time personal protective equipment compliance system for industrial sites. Live video streams
are run through a YOLOv11 detector to identify required PPE on personnel — helmets, vests, gloves,
goggles — and flag non-compliance as it happens rather than during a retrospective audit.

The throughput problem in this class of system is rarely the model; it is the camera I/O blocking the
inference loop. Decoupling capture onto its own thread so the GPU never waits on a frame read is what
makes real-time operation on live streams practical.

## Domain &amp; Techniques

| Layer | Implementation |
| :--- | :--- |
| **Detection** | YOLOv11 via Ultralytics, fine-tuned on a PPE dataset, dispatched to CUDA when available with automatic CPU fallback |
| **Threaded Capture** | `ThreadedVideoStream` reads frames continuously on a background thread with automatic source reconnection, so inference never blocks on camera I/O |
| **Compliance Logic** | `missing_ppe()` performs set difference between detected classes and the required-equipment set — a worker is non-compliant if any required item is absent |
| **Service Layer** | FastAPI exposes single-image `/detect` and live `/stream/violations` endpoints for dashboard and alerting integration |
| **Deployment** | Containerised with a slim Python base image and headless OpenCV for reproducible rollout |

## Pipeline

```
IP camera / video file
        |
        v
[ ThreadedVideoStream ]   background thread, auto-reconnect
        |                 (inference never waits on I/O)
        v
[ YOLOv11 @ CUDA ]  conf threshold filter
        |
        v
detected classes  --- set difference --->  required PPE set
        |                                        |
        v                                        v
   detections[]                            missing_ppe[]
        |                                        |
        +----------------> compliant? <----------+
                                |
                    FastAPI  /detect  |  /stream/violations
```

## Key Results

- **96.5% mAP@50** detection precision
- Multi-threaded OpenCV capture with PyTorch CUDA acceleration for real-time throughput
- Cut manual safety audit overhead by **75%**

## Quickstart

```bash
pip install -r requirements.txt

# place fine-tuned weights at models/ppe-yolov11.pt
uvicorn src.api:app --reload
```

Or with Docker:

```bash
docker build -t ppe-pipeline .
docker run -p 8000:8000 ppe-pipeline
```

## Configuration

All settings are environment-overridable:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PPE_MODEL_PATH` | `models/ppe-yolov11.pt` | Fine-tuned weights |
| `PPE_CONF_THRESHOLD` | `0.5` | Minimum detection confidence |
| `PPE_DEVICE` | `cuda` | Inference device (falls back to CPU) |

## Training

```bash
python scripts/train.py --data ppe.yaml --epochs 100
```

Fine-tunes from a YOLOv11 base checkpoint and exports to ONNX on completion.

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `src/detector.py` | `PPEDetector` — model loading, inference, compliance set logic |
| `src/video_stream.py` | `ThreadedVideoStream` — non-blocking capture with reconnection |
| `src/api.py` | FastAPI service — `/health`, `/detect`, `/stream/violations` |
| `src/config.py` | Environment-driven settings |
| `scripts/train.py` | YOLOv11 fine-tuning and ONNX export |
| `Dockerfile` | Container build |

## Project Status

**Implemented:** full inference pipeline, threaded capture, compliance evaluation, REST service and
container build.

**Note on weights:** the fine-tuned `.pt` checkpoint is not committed — model artifacts are excluded
via `.gitignore`. Train with `scripts/train.py` against a PPE dataset, or drop existing weights into
`models/`.

**Roadmap:** per-worker tracking (so a single person is not counted as multiple violations across
frames), and violation event persistence for audit reporting.

---

<div align="center">
  <sub>
    Part of the <b>AI + Robotics</b> engineering portfolio of
    <a href="https://github.com/divyansh-sachdev">Divyansh Sachdev</a><br>
    90+ national &amp; international competition wins &middot; IIT / NIT / IIIT podiums
  </sub>
</div>
