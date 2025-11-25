"""
Crowd Density Analysis Module
-----------------------------
Analyzes crowd density in stadium-like video frames by detecting people,
classifying overall density levels (LOW, MEDIUM, HIGH), and computing
density per spatial zone (2×2 grid).
"""

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Load YOLO person-detection model (pretrained)
# -----------------------------------------------------------------------------
model = YOLO("yolov8n.pt")  # Use "yolov8m.pt" for more accuracy if GPU allows


# -----------------------------------------------------------------------------
# Main density analysis function
# -----------------------------------------------------------------------------

def analyze_density(frame):
    """Analyze global and zonal crowd density in a single frame.

    Performs YOLO person detection and classifies:
      - Global crowd density level
      - People count per 2x2 grid zone
      - Zone-level density (LOW, MEDIUM, HIGH)

    Args:
        frame (np.ndarray): Input BGR video frame.

    Returns:
        dict: Dictionary containing:
            - frame_id (int): Placeholder (use external counter later).
            - people_count (int): Total persons detected.
            - density_level (str): Global density classification.
            - zones (list): List of dictionaries with zone info:
                * id (str)
                * count (int)
                * density (str)
                * rect (list): [x1, y1, x2, y2]
    """
    # YOLO prediction
    results = model.predict(frame, imgsz=640, conf=0.3, verbose=False)

    # Extract person detections (class 0)
    people = [
        box for box in results[0].boxes
        if int(box.cls[0]) == 0
    ]
    people_count = len(people)

    # ----------------------------------------------------------------------
    # Global density classification
    # ----------------------------------------------------------------------
    if people_count < 5:
        density_level = "LOW"
    elif people_count < 15:
        density_level = "MEDIUM"
    else:
        density_level = "HIGH"

    # ----------------------------------------------------------------------
    # 2×2 zone grid setup
    # ----------------------------------------------------------------------
    h, w, _ = frame.shape
    GRID_ROWS, GRID_COLS = 2, 2
    zones = []

    zone_w = w // GRID_COLS
    zone_h = h // GRID_ROWS

    # ----------------------------------------------------------------------
    # Count persons per zone (based on bounding-box center)
    # ----------------------------------------------------------------------
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            x1, y1 = j * zone_w, i * zone_h
            x2, y2 = x1 + zone_w, y1 + zone_h

            count = 0
            for box in people:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                if x1 <= cx < x2 and y1 <= cy < y2:
                    count += 1

            # Zone-level density
            zone_density = (
                "LOW" if count < 2 else
                "MEDIUM" if count < 5 else
                "HIGH"
            )

            zones.append({
                "id": f"Z{i}{j}",
                "count": count,
                "density": zone_density,
                "rect": [x1, y1, x2, y2]
            })

    # ----------------------------------------------------------------------
    # Draw visualization rectangles (optional)
    # ----------------------------------------------------------------------
    for zone in zones:
        x1, y1, x2, y2 = zone["rect"]

        color = (
            (0, 255, 0) if zone["density"] == "LOW" else
            (0, 255, 255) if zone["density"] == "MEDIUM" else
            (0, 0, 255)
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            zone["density"],
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return {
        "frame_id": 0,  # Placeholder — update externally using global counter
        "people_count": people_count,
        "density_level": density_level,
        "zones": zones
    }
