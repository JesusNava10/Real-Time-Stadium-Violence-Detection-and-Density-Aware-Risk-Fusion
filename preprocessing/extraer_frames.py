import cv2
import os

# -----------------------------------------------------------------------------
# Frame Extraction Configuration
# -----------------------------------------------------------------------------
input_video_dir = r"D:\chuch\Seguridad\Vision Project\vision-detection\videos"
output_frames_dir = r"D:\chuch\Seguridad\Vision Project\vision-detection\frames_nuevos5"
target_video = "prueba4_restaurado.mp4"

video_path = os.path.join(input_video_dir, target_video)

# Ensure output directory exists
os.makedirs(output_frames_dir, exist_ok=True)


def extract_all_frames(video_path, output_dir):
    """Extract every frame from a video while preventing duplicates.

    Args:
        video_path (str): Full path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Unable to open video: {video_path}")

    print("\n Forced frame extraction started...")

    # Load existing filenames to avoid duplicates
    existing_names = set(os.listdir(output_dir))

    frame_interval = 1  # save every frame
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"{os.path.splitext(target_video)[0]}_frame_{frame_idx:05d}.jpg"

        # Only save new frames
        if frame_name not in existing_names:
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved += 1

        frame_idx += 1

    cap.release()

    print("\n FULL extraction completed.")
    print(f" Frames saved: {saved}")


if __name__ == "__main__":
    extract_all_frames(video_path, output_frames_dir)
