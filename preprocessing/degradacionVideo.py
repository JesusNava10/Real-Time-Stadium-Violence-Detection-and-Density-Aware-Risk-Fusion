import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Axis Q6075-E PTZ Camera Simulation Parameters
# -----------------------------------------------------------------------------
axis_q6075e_params = {
    "name": "Axis Q6075-E PTZ (Simulated)",
    "resolution": (1920, 1080),
    "fps": 25,  # base reference FPS
    "exposure_alpha": 0.72,
    "exposure_beta": -35,
    "gaussian_noise_std": 10,
    "salt_pepper_prob": 0.008,
    "motion_blur_size": 9,
    "gaussian_blur_sigma": 1.4,
    "jpeg_quality": 45,
    "color_saturation": 0.9,
    "zoom_factor": 20
}


def simulate_stadium_camera(frame, params):
    """Apply stadium-style PTZ camera degradation to a single frame.

    This simulates:
      - Underexposure (darkening)
      - Gaussian noise
      - Salt-and-pepper noise
      - Motion blur
      - Gaussian blur
      - Saturation reduction
      - JPEG compression artifacts

    Args:
        frame (np.ndarray): Input BGR frame.
        params (dict): Dictionary containing degradation parameters.

    Returns:
        np.ndarray: The degraded/simulated camera frame.
    """
    # Darken/underexpose
    frame_dark = cv2.convertScaleAbs(
        frame,
        alpha=params["exposure_alpha"],
        beta=params["exposure_beta"],
    )

    # Gaussian noise
    noise = np.random.normal(
        0,
        params["gaussian_noise_std"],
        frame_dark.shape
    ).astype(np.float32)

    frame_noisy = np.clip(
        frame_dark.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    # Salt-and-pepper noise
    prob = params["salt_pepper_prob"]
    mask = np.random.rand(*frame_noisy.shape[:2])

    frame_noisy[mask < prob / 2] = 0        # black pixels
    frame_noisy[mask > 1 - prob / 2] = 255  # white pixels

    # Motion blur
    k = params["motion_blur_size"]
    kernel_motion = np.zeros((k, k))
    kernel_motion[k // 2, :] = np.ones(k) / k

    frame_motion = cv2.filter2D(frame_noisy, -1, kernel_motion)

    # Gaussian blur refinement
    frame_blur = cv2.GaussianBlur(
        frame_motion, (5, 5), params["gaussian_blur_sigma"]
    )

    # Reduce saturation (color degradation)
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * params["color_saturation"], 0, 255).astype(np.uint8)
    frame_color = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    # JPEG compression artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), params["jpeg_quality"]]
    _, encimg = cv2.imencode(".jpg", frame_color, encode_param)
    frame_compressed = cv2.imdecode(encimg, 1)

    return frame_compressed


def simulate_video(input_path, output_path, params):
    """Apply stadium camera degradation to an entire video.

    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save the simulated output video.
        params (dict): Simulation parameter dictionary.

    Raises:
        Exception: If input video cannot be opened.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Unable to open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS)

    # Use detected FPS only if valid; otherwise fallback to default
    fps = fps_read if fps_read > 5 else params["fps"]

    print(f"[INFO] Detected FPS: {fps_read:.2f} → Using: {fps:.2f}")
    print(f"[INFO] Applying simulated degradation: {params['name']}")

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        simulated = simulate_stadium_camera(frame, params)
        out.write(simulated)

        cv2.imshow("Original", frame)
        cv2.imshow(f"Simulated: {params['name']}", simulated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(
        f"[DONE] Simulation complete. {frame_count} frames processed.\n"
        f"[SAVED] Output written to: {output_path}"
    )


if __name__ == "__main__":
    simulate_video(
        "../videos/prueba4.mp4",
        "../videos/prueba4_simulado.mp4",
        axis_q6075e_params
    )
