# Garbage Sorting using YOLO and Computer Vision

## Overview

This project presents an AI-based garbage sorting system that uses deep learning object detection models to classify waste materials automatically. The system is designed to support automated segregation in recycling facilities, smart city waste management, and robotic sorting applications.

The model is trained using a custom dataset prepared with Roboflow and utilizes Ultralytics YOLO architectures for accurate and real-time detection.

---

## Objectives

* Automate waste classification using computer vision and deep learning
* Improve efficiency and accuracy of recycling processes
* Reduce manual sorting efforts
* Enable integration with automated conveyor or robotic sorting systems

---

## Technologies Used

* Python
* Ultralytics YOLO (YOLOv8, YOLOv11)
* OpenCV
* Roboflow (Dataset Annotation and Management)
* NumPy
* Matplotlib

---

## Project Structure

```
code/
│
├── train.py              # Model training script
├── t4.py                 # Inference / testing script
├── data.yaml             # Dataset configuration file
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore rules
```

---

## System Workflow

1. Collect waste images using cameras or public datasets
2. Annotate and organize the dataset using Roboflow
3. Export annotated dataset in YOLO format
4. Train YOLO-based detection model using custom dataset
5. Evaluate model performance using validation data
6. Perform inference on test images or live camera feed
7. (Optional) Integrate detection output with sorting mechanisms such as robotic arms or conveyor systems

---

## Installation and Setup

### Install Dependencies

```
pip install -r requirements.txt
```

---

## Model Training

```
python train.py
```

---

## Model Inference

```
python t4.py
```

---

## Model Configuration

* Architecture: YOLOv8n / YOLOv11n
* Task Type: Object Detection
* Dataset Source: Custom Garbage Dataset (Roboflow)
* Example Classes: Plastic, Paper, Metal, Organic
* Input Resolution: 640 × 640
* Optimization: Adam Optimizer
* Loss Functions: CIoU Loss, Classification Loss, Objectness Loss

---

## Applications

* Automated recycling plants
* Smart waste segregation systems
* Industrial sorting automation
* Robotics-based waste handling systems

---

## Future Enhancements

* Real-time webcam-based detection
* Edge deployment using embedded platforms such as Raspberry Pi or NVIDIA Jetson
* Expansion of waste classification categories
* Integration with industrial automation dashboards

---

## Author

Saumitra Chaporkar
Automation and Robotics Engineering

---

## License

This project is intended for academic and research purposes.
