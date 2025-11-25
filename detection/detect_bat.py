"""
Bat Detection Module
--------------------
Detects 'bat' objects in frames using a trained YOLO model.
Provides bounding boxes and confidence scores.
"""

import os
import cv2
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Load YOLO model
# -----------------------------------------------------------------------------

# Absolute path of the current directory (detection/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up (vision-detection/)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Path to trained YOLO model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "detection",
    "runs",
    "train",
    "violencia_estadio_yolov10s_v13",
    "weights",
    "best.pt"
)
MODEL_PATH = os.path.abspath(MODEL_PATH)

print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")

# Load YOLO model
model = YOLO(MODEL_PATH)


# -----------------------------------------------------------------------------
# Main detection function
# -----------------------------------------------------------------------------

def detect_bat(frame):
    """Detect a 'bat' object inside a frame.

    Uses a YOLO model to predict objects and checks if class 'bat'
    appears in the results. Returns a dictionary containing:

        - bat_detected (bool): True if a bat was detected.
        - confidence (float): Confidence score of the detection.

    Args:
        frame (np.ndarray): Input BGR frame.

    Returns:
        dict: Detection results.
    """
    # Recommended parameters for detecting small/thin bats:
    # conf=0.12 → captures more candidates
    # iou=0.45 → reduces overlapping detections
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.12,
        verbose=False
    )

    bat_detected = False
    confidence = 0.0

    # Iterate over detected boxes
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        if label.lower() == "bat":
            bat_detected = True
            confidence = conf

            # Draw bounding box (red)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Bat {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    return {
        "bat_detected": bat_detected,
        "confidence": confidence
    }


# -----------------------------------------------------------------------------
# Standalone test (quick run)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    video_path = os.path.join(BASE_DIR, "videos", "prueba4_restaurado.mp4")
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = os.path.join(BASE_DIR, "videos", "prueba_detectado4_custom.mp4")
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_bat(frame)
        out.write(frame)

        cv2.imshow("Bat Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[DONE] Detection video saved at: {out_path}")
