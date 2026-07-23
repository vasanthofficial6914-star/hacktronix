"""
Video Stream Provider & Synthetic Frame Generator.

Provides live camera frame capture, synthetic dynamic video stream generation,
and frame annotation (drawing bounding boxes, labels, and knowledge overlays).
"""

import time
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List


class VideoStreamManager:
    """
    Manages camera/video streams and renders annotated visual overlays.
    """

    def __init__(self, source: int = 0) -> None:
        self.source = source
        self.cap = None

    def open_camera(self) -> bool:
        """Attempts to open local webcam or video device."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            return self.cap.isOpened()
        except Exception:
            self.cap = None
            return False

    def get_frame(self) -> np.ndarray:
        """
        Retrieves frame from camera. If camera is unavailable, generates synthetic visual scene.
        """
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame

        # Generate synthetic frame with moving shapes
        return self.generate_synthetic_frame()

    def generate_synthetic_frame(self) -> np.ndarray:
        """
        Generates dynamic 640x480 synthetic frame representing a room scene with objects.
        """
        frame = np.full((480, 640, 3), (25, 30, 45), dtype=np.uint8)

        # Draw simulated background room grid
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (40, 45, 60), 1)
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (40, 45, 60), 1)

        # Moving object simulation using sinusoidal animation
        t = time.time()
        c_x = int(320 + 100 * np.sin(t))
        c_y = int(240 + 50 * np.cos(t))

        # Simulated Laptop / Box
        cv2.rectangle(frame, (220, 180), (420, 320), (180, 120, 50), -1)
        cv2.putText(frame, "SIMULATED WORKSTATION", (225, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Dynamic Object (Mug/Robot)
        cv2.circle(frame, (c_x, c_y), 30, (50, 180, 90), -1)

        # Live Timestamp banner
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"LIVE STREAM - {ts_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: Dict[str, Any]) -> np.ndarray:
        """
        Annotates camera frame with bounding boxes, colors, labels, and confidence.
        """
        annotated = frame.copy()

        # Draw objects
        all_items = detections.get("objects", []) + detections.get("people", []) + detections.get("faces", [])
        for item in all_items:
            bbox_dict = item.get("bounding_box", {})
            if not bbox_dict:
                continue

            xmin = int(bbox_dict["xmin"])
            ymin = int(bbox_dict["ymin"])
            xmax = int(bbox_dict["xmax"])
            ymax = int(bbox_dict["ymax"])

            cat = item.get("category", "object")
            color = (0, 255, 0)  # Green for object
            if cat == "person":
                color = (0, 0, 255)  # Red for person
            elif cat == "face":
                color = (255, 255, 0)  # Yellow for face

            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            label_text = f"{item['name']} ({item['confidence']:.2f})"
            cv2.putText(annotated, label_text, (xmin, max(15, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated

    def release(self) -> None:
        """Releases camera hardware resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
