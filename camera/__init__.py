"""RealSense acquisition boundary.

Importing :mod:`camera` does not import ``pyrealsense2``.  The vendor SDK is
loaded lazily only when a live or bag source is started.
"""

from .bag_source import BagSource, RealSenseBagSource
from .device_diagnostics import DeviceDiagnostics, StreamDiagnostics
from .frame_source import (
    BagEndOfStream,
    BagPlaybackError,
    CameraConfigurationError,
    CameraConnectionError,
    CameraError,
    CameraFrame,
    CameraIntrinsics,
    FrameAcquisitionError,
    FrameAlignmentError,
    FrameShapeMismatchError,
    FrameSource,
    FrameTimeoutError,
    RealSenseUnavailableError,
    SourceNotStartedError,
    UnsupportedDeviceError,
)
from .realsense_camera import RealSenseCamera

__all__ = [
    "BagEndOfStream",
    "BagPlaybackError",
    "BagSource",
    "CameraConfigurationError",
    "CameraConnectionError",
    "CameraError",
    "CameraFrame",
    "CameraIntrinsics",
    "DeviceDiagnostics",
    "FrameAcquisitionError",
    "FrameAlignmentError",
    "FrameShapeMismatchError",
    "FrameSource",
    "FrameTimeoutError",
    "RealSenseBagSource",
    "RealSenseCamera",
    "RealSenseUnavailableError",
    "SourceNotStartedError",
    "StreamDiagnostics",
    "UnsupportedDeviceError",
]
