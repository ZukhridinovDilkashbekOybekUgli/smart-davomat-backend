# Smart Davomat AI — Lightweight Backend

This version is designed to avoid the previous Render build failure.
It removes dlib and the full PyTorch/Ultralytics runtime.

Features:
- Browser camera
- OpenCV Haar face detection
- OpenCV LBPH face recognition
- Student enrollment
- Attendance API + SQLite
- YOLOv8n ONNX phone detection through OpenCV DNN
- `/api/health`

Render settings:
- Language: Docker
- Root Directory: blank
- Dockerfile Path: ./Dockerfile
- Docker Command: blank
- Health Check Path: /api/health

Note: LBPH score is an MVP similarity score, not a calibrated probability.
