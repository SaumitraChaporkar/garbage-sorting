# Garbage Sorting using YOLO & Computer Vision

## Overview

This project implements an **AI-powered garbage sorting system** using state-of-the-art object detection models (YOLOv8 and YOLOv11). The system is trained on a **custom Roboflow dataset** to detect and classify different types of waste for automated segregation in recycling plants, smart cities, and robotic sorting stations.

The goal is to reduce human effort, improve recycling efficiency, and enable real-time intelligent waste classification.

---

## Key Objectives

* Automate garbage classification using deep learning
* Reduce manual waste segregation effort
* Improve recycling accuracy and throughput
* Enable real-time detection for robotic or conveyor-belt systems

---

## Tech Stack

* **Python**
* **Ultralytics YOLO (YOLOv8, YOLOv11)**
* **OpenCV**
* **Roboflow** (dataset collection & annotation)
* **NumPy, Matplotlib**

---

## Project Structure

```
code/
│
├── train.py              # YOLO training script
├── t4.py                 # Inference / testing script
├── data.yaml             # Dataset configuration file
├── README.dataset        # Roboflow dataset metadata
├── README.roboflow       # Roboflow project metadata
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore rules
```

---

## System Workflow

1. Collect garbage images using camera or open datasets
2. Annotate images using Roboflow
3. Export dataset in YOLO format
4. Train YOLOv8/YOLOv11 model on custom dataset
5. Validate model performance
6. Run inference on test images or live camera feed
7. (Optional) Trigger robotic arm or sorting actuator

---

## How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python train.py
```

### 3️⃣ Run Inference

```bash
python t4.py
```

---

## Model Details

* **Architecture:** YOLOv8n / YOLOv11n
* **Task:** Object Detection
* **Dataset:** Custom garbage dataset (Roboflow)
* **Classes:** Plastic, Paper, Metal, Organic (example)
* **Input Size:** 640 × 640
* **Optimizer:** Adam
* **Loss:** CIoU + Classification + Objectness

---

## Performance (Example)

| Metric        | Value |
| ------------- | ----- |
| mAP@0.5       | XX %  |
| Precision     | XX %  |
| Recall        | XX %  |
| Inference FPS | XX    |

*(Replace XX with your real results if available.)*

---

## Demo

* Add screenshots of predictions
* Add YOLO result images
* Add short demo video link (Google Drive / YouTube)

---

## Future Improvements

* Real-time webcam integration
* Edge deployment on Raspberry Pi / Jetson Nano
* Robotic arm integration for physical sorting
* Additional waste categories
* Web dashboard for monitoring

---

## Resume Bullet Points

* Built an AI-based garbage sorting system using YOLOv8 and YOLOv11 for automated waste classification
* Trained a custom object detection model on Roboflow-annotated garbage dataset
* Implemented real-time inference pipeline using Python and OpenCV
* Designed a reproducible ML workflow with clean Git version control

---

## Author

**Saumitra Chaporkar**
Automation & Robotics Engineer
Robotics | AI | Computer Vision

---

## If you like this project

Give this repository a  on GitHub to support my work!
