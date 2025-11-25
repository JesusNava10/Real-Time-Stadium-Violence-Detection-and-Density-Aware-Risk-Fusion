"""
main_pipeline.py
Real-Time Stadium Violence Detection Pipeline

This script runs the full detection and alert pipeline:
1. Dangerous object detection (bat) using YOLOv10 custom model.
2. Crowd density estimation using YOLOv8n.
3. Global risk fusion (SAFE / LOW / MEDIUM / HIGH / CRITICAL).
4. CSV logging + annotated video output + risk dashboard window.
"""

import cv2
import os
import csv
import numpy as np

from alerts.alert_manager import evaluate_alerts
from detection.detect_bat import detect_bat
from crowd_counting.crowd_density_zones import analyze_density


def draw_global_risk_window(risk_level, bat_count, density_level):
    """Draws a separate dashboard window showing the global system risk.

    Args:
        risk_level (str): Current global risk level.
        bat_count (int): Number of total bat detections.
        density_level (str): Current density level (LOW/MEDIUM/HIGH).

    Returns:
        np.ndarray: Panel rendered as an image.
    """
    panel = np.zeros((250, 400, 3), dtype=np.uint8)

    # Risk color mapping
    colors = {
        "SAFE": (0, 255, 0),
        "MEDIUM": (0, 165, 255),
        "HIGH": (0, 0, 255),
        "CRITICAL": (0, 0, 128)
    }
    color = colors.get(risk_level, (255, 255, 255))

    # Background color
    panel[:] = (30, 30, 30)

    # Title
    cv2.putText(panel, "RISK MONITOR", (70, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Risk level
    cv2.putText(panel, f"Level: {risk_level}", (80, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    # Stats
    cv2.putText(panel, f"Bat detections: {bat_count}", (60, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(panel, f"Density: {density_level}", (60, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return panel


def main():
    """Runs the complete video-processing pipeline: detection, density, risk fusion, logging, and visualization."""
    video_path = os.path.abspath("videos/prueba4_restaurado.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    # Output video configuration
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = "videos/prueba1_alertas.mp4"

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("🎥 Processing video...")

    # CSV Logging setup
    log_path = "data_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "people_count",
            "bat_detected",
            "bat_confidence",
            "bat_total_detections",
            "density_level",
            "global_risk"
        ])

    frame_id = 0
    bat_detected_count = 0
    last_bat_frame = -30
    global_risk = "SAFE"

    # ========================== MAIN LOOP ==========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # ----- Detection modules -----
        detection_state = detect_bat(frame)
        crowd_state = analyze_density(frame)
        alert_state, frame = evaluate_alerts(frame, detection_state, crowd_state)

        # ----- Bat detection counting + global risk logic -----
        if detection_state["bat_detected"]:
            # Avoid counting repeated detections in consecutive frames
            if frame_id - last_bat_frame > 30:
                bat_detected_count += 1
                last_bat_frame = frame_id
                print(f"🚨 Bat event #{bat_detected_count}")

                # Global risk escalation
                if bat_detected_count == 1:
                    global_risk = "MEDIUM"
                elif bat_detected_count == 2:
                    global_risk = "HIGH"
                elif bat_detected_count >= 3:
                    global_risk = "CRITICAL"

                cv2.imwrite(f"vision-detection/alert_frame_{bat_detected_count}.jpg", frame)

        # De-escalate risk if long time has passed without detections
        if not detection_state["bat_detected"] and frame_id - last_bat_frame > 150:
            global_risk = "SAFE"

        # ----- Risk dashboard -----
        density_level = crowd_state.get("density_level", "LOW")
        risk_panel = draw_global_risk_window(global_risk, bat_detected_count, density_level)

        # ----- Save CSV data -----
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                crowd_state.get("people_count", 0),
                detection_state["bat_detected"],
                detection_state["confidence"],
                bat_detected_count,
                density_level,
                global_risk
            ])

        # ----- Visualization -----
        cv2.imshow("Alert System", frame)
        cv2.imshow("Global Risk Monitor", risk_panel)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Output video saved at: {out_path}")


if __name__ == "__main__":
    main()
