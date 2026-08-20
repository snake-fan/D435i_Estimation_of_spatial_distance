"""QR-code detection and optional sub-pixel corner refinement."""

from .corner_refinement import refine_qr_corners
from .qr_detector import QRDetection, QRDetector

__all__ = ["QRDetection", "QRDetector", "refine_qr_corners"]
