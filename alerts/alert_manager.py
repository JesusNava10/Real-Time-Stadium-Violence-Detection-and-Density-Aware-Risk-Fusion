"""
Alert Evaluation Module
-----------------------
Combines bat-detection results and crowd-density analysis
to determine the overall risk level for a given frame.
"""

import cv2


def evaluate_alerts(frame, detection_state, crowd_state, show_visual=True):
    """Evaluate risk level based on bat detection and crowd density.

    Combines:
      - Dangerous object detection (bat)
      - Crowd density classification
    to determine one of the following states:
      * CRITICAL
      * MEDIUM
      * LOW
      * SAFE

    Args:
        frame (np.ndarray): Frame where alerts will be drawn.
        detection_state (dict): Output from detect_bat().
        crowd_state (dict): Output from analyze_density().
        show_visual (bool): Whether to draw alert text on the frame.

    Returns:
        tuple:
            dict: Alert state containing:
                - frame_id (int)
                - risk_level (str)
                - message (str)
            np.ndarray: Frame with visual overlays (optional).
    """
    bat_detected = detection_state.get("bat_detected", False)
    density_level = crowd_state.get("density_level", "LOW")

    # ----------------------------------------------------------------------
    # Risk evaluation logic
    # ----------------------------------------------------------------------
    if bat_detected and density_level == "HIGH":
        risk_level = "CRITICAL"
        message = "Bat detected in high-density zone"
        color = (0, 0, 255)  # Red
    elif bat_detected and density_level in ["MEDIUM", "LOW"]:
        risk_level = "MEDIUM"
        message = "Bat detected in low-density zone"
        color = (0, 165, 255)  # Orange
    elif not bat_detected and density_level == "HIGH":
        risk_level = "LOW"
        message = "High density with no dangerous object detected"
        color = (0, 255, 255)  # Yellow
    else:
        risk_level = "SAFE"
        message = "No risk detected"
        color = (0, 255, 0)  # Green

    # ----------------------------------------------------------------------
    # Optional visual overlay
    # ----------------------------------------------------------------------
    if show_visual and frame is not None:
        cv2.putText(
            frame,
            f"Risk: {risk_level}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )
        cv2.putText(
            frame,
            message,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ----------------------------------------------------------------------
    # Output dictionary
    # ----------------------------------------------------------------------
    alert_state = {
        "frame_id": crowd_state.get("frame_id", -1),
        "risk_level": risk_level,
        "message": message,
    }

    return alert_state, frame
