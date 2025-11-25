"""
Video Restoration Pipeline (Stage 2)
------------------------------------
Restores degraded stadium-camera footage by applying the same enhancement
pipeline used in Experiment 3, including color correction, denoising,
contrast recovery, and multi-stage sharpening.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def restore_frame(frame):
    """Apply advanced restoration to a single frame.

    This pipeline is optimized for recovering thin objects (e.g., bats)
    by enhancing color constancy, brightness, and fine-edge sharpness.

    Args:
        frame (np.ndarray): Input BGR frame.

    Returns:
        np.ndarray: Restored BGR frame.
    """
    # ----------------------------------------------------------------------
    # LAB white balance correction
    # ----------------------------------------------------------------------
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a, avg_b = np.average(lab[:, :, 1]), np.average(lab[:, :, 2])

    lab[:, :, 1] -= (avg_a - 128) * (lab[:, :, 0] / 255.0)
    lab[:, :, 2] -= (avg_b - 128) * (lab[:, :, 0] / 255.0)

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    img_white = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ----------------------------------------------------------------------
    # Balanced color denoising
    # ----------------------------------------------------------------------
    img_denoised = cv2.fastNlMeansDenoisingColored(
        img_white, None, 3, 3, 7, 21
    )

    # ----------------------------------------------------------------------
    # Mild gamma correction to recover midtones
    # ----------------------------------------------------------------------
    gamma = 1.05
    img_gamma = np.power(img_denoised / 255.0, gamma)
    img_gamma = np.uint8(np.clip(img_gamma * 255, 0, 255))

    # ----------------------------------------------------------------------
    # Perceptual enhancement (PIL: Color → Contrast → Sharpen)
    # ----------------------------------------------------------------------
    pil_img = Image.fromarray(cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB))
    img_color = ImageEnhance.Color(pil_img).enhance(1.08)
    img_contrast = ImageEnhance.Contrast(img_color).enhance(1.25)
    img_sharp = ImageEnhance.Sharpness(img_contrast).enhance(1.35)

    img_pil = cv2.cvtColor(np.array(img_sharp), cv2.COLOR_RGB2BGR)

    # ----------------------------------------------------------------------
    # High-pass sharpening for edge recovery
    # ----------------------------------------------------------------------
    blur = cv2.GaussianBlur(img_pil, (0, 0), 3)
    high_pass = cv2.addWeighted(img_pil, 1.6, blur, -0.6, 0)

    # ----------------------------------------------------------------------
    # Final unsharp mask for micro-contrast
    # ----------------------------------------------------------------------
    gaussian = cv2.GaussianBlur(high_pass, (0, 0), 1.0)
    img_unsharp = cv2.addWeighted(high_pass, 1.4, gaussian, -0.4, 0)

    # ----------------------------------------------------------------------
    # Final brightness adjustment in HSV
    # ----------------------------------------------------------------------
    hsv = cv2.cvtColor(img_unsharp, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * 1.08, 0, 255).astype(np.uint8)

    img_final = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    return img_final


def restore_video(input_path, output_path):
    """Restore an entire video frame-by-frame.

    Args:
        input_path (str): Path to the input degraded video.
        output_path (str): Path where the restored video will be saved.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Unable to open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS)

    # If FPS is invalid or too low, fall back to 25 fps
    fps = fps_read if fps_read > 5 else 25
    print(f"Restoring degraded video... (detected FPS: {fps_read:.2f}, used: {fps:.2f})")

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        restored = restore_frame(frame)
        out.write(restored)

        cv2.imshow("Degraded", frame)
        cv2.imshow("Restored", restored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Restoration complete ({frame_count} frames). Saved at: {output_path}")


if __name__ == "__main__":
    restore_video(
        "../videos/prueba4_simulado.mp4",
        "../videos/prueba4_restaurado.mp4"
    )
