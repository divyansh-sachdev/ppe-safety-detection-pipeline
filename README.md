# PPE Safety Detection Pipeline

Real-time personal protective equipment (PPE) compliance detection built on YOLOv11, with a multi-threaded OpenCV video pipeline and CUDA-accelerated PyTorch inference.

## Tech Stack
Python · PyTorch · YOLOv11 · OpenCV · FastAPI · Docker

## Key Results
- Achieved **96.5% mAP@50** detection precision
- Multi-threaded OpenCV video streams + PyTorch CUDA acceleration for real-time throughput
- Cut manual safety audit overhead by **75%**

## Overview
The pipeline ingests live or recorded video streams, runs YOLOv11-based detection to identify required PPE (helmets, vests, gloves, etc.) on personnel, and flags non-compliance in real time. A FastAPI service exposes detection results for downstream alerting/dashboarding, and the whole stack is containerized with Docker for deployment.
