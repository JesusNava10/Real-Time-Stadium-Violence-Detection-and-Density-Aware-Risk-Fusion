"""
Crowd Counting Module
---------------------
Performs person detection on a video using a pretrained YOLO model.
Draws bounding boxes for each detected person and displays the total count.
"""

import cv2
from ultralytics import YOLO


def run_crowd_counting(video_path: str):
    """Run crowd counting on a video using a YOLO model.

    Detects persons in each frame, draws bounding boxes, and overlays
    the total number of detected people.

    Args:
        video_path (str): Path to the input video.
    """
    # Load pretrained YOLO model for person detection
    model = YOLO("yolov8n.pt")  # Use "yolov8m.pt" for higher accuracy

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Unable to open the video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        person_count = 0

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                person_count += 1

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display person count on frame
        cv2.putText(
            frame,
            f"Detected persons: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("Crowd Counting", frame)

        # Press 'Q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "../videos/prueba1_restaurado.mp4"
    run_crowd_counting(video_path)
