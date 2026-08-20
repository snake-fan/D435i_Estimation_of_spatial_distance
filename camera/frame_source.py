"""SDK-neutral frame-source contracts used by the measurement pipeline.

The rest of the project may import this module on machines that do not have
``pyrealsense2`` installed.  Native SDK objects carried by :class:`CameraFrame`
are deliberately typed as ``object`` and are optional; consumers should use
the NumPy images and the calibration/deprojection methods on ``FrameSource``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class CameraError(RuntimeError):
    """Base class for all camera-layer failures."""


class RealSenseUnavailableError(CameraError):
    """Raised when a RealSense source is used without ``pyrealsense2``."""


class CameraConfigurationError(CameraError):
    """Raised when a requested stream or source configuration is invalid."""


class CameraConnectionError(CameraError):
    """Raised when a camera or bag pipeline cannot be started."""


class UnsupportedDeviceError(CameraConnectionError):
    """Raised when the selected live device is not supported by this project."""


class SourceNotStartedError(CameraError):
    """Raised when data is requested before a source has been started."""


class FrameAcquisitionError(CameraError):
    """Raised when a source fails to provide a usable synchronized frameset."""


class FrameTimeoutError(FrameAcquisitionError):
    """Raised when no frameset arrives within the configured timeout."""


class FrameAlignmentError(FrameAcquisitionError):
    """Raised when Left IR cannot be mapped safely into the Depth viewport."""


class FrameShapeMismatchError(FrameAlignmentError):
    """Raised when the aligned IR and native Depth arrays have different shapes."""


class BagPlaybackError(CameraError):
    """Raised for a malformed or unusable RealSense bag recording."""


class BagEndOfStream(BagPlaybackError, EOFError):
    """Raised when a non-repeating bag reaches its end.

    This distinct exception lets the application treat normal bag EOF as a
    control-flow event instead of parsing an SDK-specific RuntimeError string.
    """


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Serializable, SDK-neutral pinhole calibration metadata.

    Deprojection should still be performed through :meth:`FrameSource.deproject_pixel`
    so the RealSense SDK can apply its own distortion model.  This structure is
    primarily for diagnostics, validation, and consumers that need ``fx/fy``.
    """

    width: int
    height: int
    ppx: float
    ppy: float
    fx: float
    fy: float
    distortion_model: str
    coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("intrinsic image dimensions must be positive")
        values = (self.ppx, self.ppy, self.fx, self.fy, *self.coefficients)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("intrinsic parameters must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("intrinsic focal lengths must be positive")

    @property
    def coeffs(self) -> tuple[float, ...]:
        """Compatibility alias matching ``pyrealsense2.intrinsics.coeffs``."""

        return self.coefficients

    @property
    def model(self) -> str:
        """Compatibility alias matching ``pyrealsense2.intrinsics.model``."""

        return self.distortion_model

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "fx": self.fx,
            "fy": self.fy,
            "distortion_model": self.distortion_model,
            "coefficients": list(self.coefficients),
        }


@dataclass(slots=True)
class CameraFrame:
    """One synchronized frame in the native Depth pixel coordinate system.

    ``ir_image`` is Left IR (stream index 1) mapped into the native Depth
    viewport.  ``depth_image`` remains the native, unresampled Z16 image.  The
    two arrays therefore have the same height and width and their pixel
    coordinates may be used together.
    """

    ir_image: np.ndarray
    depth_image: np.ndarray
    timestamp_ms: float
    frame_number: int
    depth_frame: object | None = field(default=None, repr=False, compare=False)
    ir_frame: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ir_image, np.ndarray):
            raise TypeError("ir_image must be a numpy.ndarray")
        if not isinstance(self.depth_image, np.ndarray):
            raise TypeError("depth_image must be a numpy.ndarray")
        if self.ir_image.ndim != 2 or self.depth_image.ndim != 2:
            raise ValueError("IR and Depth images must both be two-dimensional")
        if self.ir_image.shape != self.depth_image.shape:
            raise FrameShapeMismatchError(
                "aligned Left IR shape "
                f"{self.ir_image.shape} does not match native Depth shape "
                f"{self.depth_image.shape}"
            )
        if not np.isfinite(self.timestamp_ms):
            raise ValueError("frame timestamp must be finite")
        if self.frame_number < 0:
            raise ValueError("frame number cannot be negative")


class FrameSource(ABC):
    """Common contract shared by a live D435i and RealSense bag playback."""

    @abstractmethod
    def start(self) -> None:
        """Start the source and make calibration information available."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the source.  Implementations must make repeated calls safe."""

    @abstractmethod
    def get_frames(self, timeout_ms: int | None = None) -> CameraFrame:
        """Return the next synchronized, Depth-viewport frame."""

    @abstractmethod
    def get_depth_intrinsics(self) -> CameraIntrinsics:
        """Return intrinsics for the native Depth viewport."""

    @abstractmethod
    def get_depth_scale(self) -> float:
        """Return metres per integer unit in ``CameraFrame.depth_image``."""

    @abstractmethod
    def deproject_pixel(
        self,
        pixel: tuple[float, float] | list[float] | np.ndarray,
        depth_m: float,
    ) -> np.ndarray:
        """Deproject one Depth-viewport pixel through the vendor calibration.

        The returned three-vector is in metres in the native Depth camera
        coordinate system.
        """

    def deproject_pixel_to_point(
        self,
        pixel: tuple[float, float] | list[float] | np.ndarray,
        depth_m: float,
    ) -> np.ndarray:
        """Verbose compatibility alias for :meth:`deproject_pixel`."""

        return self.deproject_pixel(pixel, depth_m)

    @property
    @abstractmethod
    def started(self) -> bool:
        """Whether this source currently owns a running pipeline."""

    @property
    def is_eof(self) -> bool:
        """Whether a finite source is known to be exhausted."""

        return False

    def __enter__(self) -> FrameSource:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
