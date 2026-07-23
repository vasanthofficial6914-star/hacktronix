"""
Unit Tests for Module 4: Vision & Object Detection Engine.
"""

import numpy as np
import pytest
from hacktronix.infrastructure.vision.detector import OpenCVVisionDetector
from hacktronix.infrastructure.vision.stream import VideoStreamManager


def test_synthetic_video_frame_generation():
    stream_mgr = VideoStreamManager()
    frame = stream_mgr.generate_synthetic_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)


def test_vision_detector_with_frame():
    detector = OpenCVVisionDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a white box in middle
    frame[100:200, 100:200] = 255

    results = detector.detect_objects_and_faces(frame)
    assert "objects" in results
    assert "people" in results
    assert "faces" in results
    assert results["frame_size"] == {"width": 640, "height": 480}


def test_draw_detections_on_frame():
    stream_mgr = VideoStreamManager()
    frame = stream_mgr.generate_synthetic_frame()

    detections = {
        "objects": [
            {
                "name": "Test Box",
                "category": "object",
                "confidence": 0.9,
                "bounding_box": {"xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150},
            }
        ]
    }
    annotated = VideoStreamManager.draw_detections(frame, detections)
    assert annotated is not None
    assert annotated.shape == frame.shape
