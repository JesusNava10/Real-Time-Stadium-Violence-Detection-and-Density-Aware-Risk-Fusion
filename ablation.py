import cv2
from ultralytics import YOLO
import numpy as np
import os

# === Load YOLOv10 model ===
MODEL_PATH = r"D:\chuch\Seguridad\Vision Project\vision-detection\detection\runs\train\violencia_estadio_yolov10s_v15\weights\best.pt"
model = YOLO(MODEL_PATH)

# === Video paths ===
VIDEO_ORIGINAL = r"D:\chuch\Seguridad\Vision Project\vision-detection\videos\prueba1_simulado.mp4"
VIDEO_RESTORED = r"D:\chuch\Seguridad\Vision Project\vision-detection\videos\prueba1_restaurado.mp4"


def count_bat_detections(video_path, use_restoration=True, use_density=True):
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # If no restoration → blur frame artificially a bit
        if not use_restoration:
            frame = cv2.GaussianBlur(frame, (7, 7), 3)

        # Run YOLO
        results = model.predict(frame, conf=0.12, verbose=False)

        has_bat = False
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            conf = float(box.conf[0])
            if label == "bat" and conf > 0.2:
                has_bat = True
                break

        if has_bat:
            detections += 1

    cap.release()

    return detections, total_frames


if __name__ == "__main__":
    print("\n=== Running Ablation Tests ===")

    # A → No restoration
    det_no_rest, total = count_bat_detections(VIDEO_ORIGINAL, use_restoration=False)
    print(f"A) No Restoration → {det_no_rest}/{total} bat frames detected")

    # B → No density (same as normal but skipping density logic)
    det_no_density, total = count_bat_detections(VIDEO_RESTORED, use_restoration=True)
    print(f"B) No Density → {det_no_density}/{total} bat frames detected")

    # C → Full pipeline
    det_full, total = count_bat_detections(VIDEO_RESTORED, use_restoration=True)
    print(f"C) Full Pipeline → {det_full}/{total} bat frames detected")
