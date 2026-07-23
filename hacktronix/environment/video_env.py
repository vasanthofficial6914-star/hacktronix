"""
Video Environment Wrapper for HackModel-AI.

Provides a unified interface for camera or synthetic video stream
integrated with the Vision World Modeler pipeline.
"""

from typing import Any, Dict
import numpy as np

from hacktronix.infrastructure.vision.stream import VideoStreamManager
from hacktronix.infrastructure.vision.detector import OpenCVVisionDetector


class VideoEnvironment:
    """
    Wraps the VideoStreamManager and VisionDetector for the Track 2 observation loop.
    """

    def __init__(self, source: int = 0, use_camera: bool = False) -> None:
        self.stream = VideoStreamManager(source)
        self.detector = OpenCVVisionDetector()
        self.use_camera = use_camera
        if use_camera:
            self.stream.open_camera()

    def get_observation(self) -> Dict[str, Any]:
        """
        Capture frame, run detection, and return structured observation dict.
        """
        frame = self.stream.get_frame()
        detections = self.detector.detect_objects_and_faces(frame)
        annotated = VideoStreamManager.draw_detections(frame, detections)
        return {
            "frame": frame,
            "annotated_frame": annotated,
            "detections": detections,
        }

    def release(self) -> None:
        """Release resources."""
        self.stream.release()
