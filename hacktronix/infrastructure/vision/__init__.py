"""
Vision Detector Infrastructure package.
"""

from hacktronix.infrastructure.vision.detector import OpenCVVisionDetector
from hacktronix.infrastructure.vision.stream import VideoStreamManager

__all__ = ["OpenCVVisionDetector", "VideoStreamManager"]
