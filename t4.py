import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import os

# -----------------------------------
# ✅ CONFIGURATION
# -----------------------------------
MODEL_PATH = r"D:\garbge project\segmentacion de residuos copy.v1i.yolov8-obb\runs\detect\garbage_training_lowmem\weights\best.pt"
CAMERA_INDEX = 0  # Change to 1 if external USB camera
CONF_THRESHOLD = 0.6
OUTPUT_VIDEO_PATH = r"D:\garbge project\sorting_demo_output.mp4"

# -----------------------------------
# ✅ LOAD YOLO MODEL
# -----------------------------------
model = YOLO(MODEL_PATH)
VALID_CLASSES = ["paper", "plastic", "other"]  # Detection classes

# -----------------------------------
# ✅ CAMERA SETUP
# -----------------------------------
cap = cv2.VideoCapture("http://192.168.137.52:8080//video")
if not cap.isOpened():
    print("❌ Error: Cannot access camera.")
    exit()

# Get camera properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))
if fps_video == 0:  # fallback
    fps_video = 15

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_video, (frame_width, frame_height))

print("✅ Camera connected! Recording started.")
print("🎥 Press 'q' to stop recording and close.")
prev_time = time.time()
pred_buffer = deque(maxlen=10)

# -----------------------------------
# ✅ BACKGROUND NORMALIZATION
# -----------------------------------
def normalize_background(img):
    """Neutralize lighting and background variations."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(img, (5, 5), 0)

# -----------------------------------
# ✅ MAIN LOOP
# -----------------------------------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received.")
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # skip alternate frames for better FPS

    # Normalize background
    frame_norm = normalize_background(frame)

    # YOLO prediction
    results = model.predict(frame_norm, conf=CONF_THRESHOLD, verbose=False)
    current_preds = []

    # FPS calc
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls] if cls in model.names else "unknown"
            if label not in VALID_CLASSES:
                label = "other"

            conf = float(box.conf[0])
            current_preds.append(label)

            (x1, y1, x2, y2) = map(int, box.xyxy[0])
            color = (0, 255, 0) if label == "plastic" else (255, 255, 255) if label == "paper" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if current_preds:
        pred_buffer.extend(current_preds)
        stable_pred = max(set(pred_buffer), key=pred_buffer.count)
        cv2.putText(frame, f"Detected: {stable_pred}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    # Add FPS & logo text
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Smart Waste Sorting System", (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # ✅ Save frame to video
    out.write(frame)

    # ✅ Show live feed
    cv2.imshow("MVS Sorting System - Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Recording stopped by user.")
        break

# -----------------------------------
# ✅ CLEANUP
# -----------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"🎞️ Video saved successfully at: {OUTPUT_VIDEO_PATH}")
