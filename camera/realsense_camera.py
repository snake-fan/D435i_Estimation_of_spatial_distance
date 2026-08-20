"""Live Intel RealSense D435i source.

``pyrealsense2`` is imported only when :meth:`RealSenseCamera.start` is called.
Importing this module therefore remains safe in development and CI environments
without a RealSense SDK installation.
"""

from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .device_diagnostics import DeviceDiagnostics, StreamDiagnostics
from .frame_source import (
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

LOGGER = logging.getLogger(__name__)


def _require_realsense() -> Any:
    """Load ``pyrealsense2`` on demand with an actionable error message."""

    try:
        return importlib.import_module("pyrealsense2")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RealSenseUnavailableError(
            "pyrealsense2 is required only for live camera or .bag use. "
            "Install a RealSense SDK/Python binding compatible with this "
            "machine and interpreter before starting a camera source."
        ) from exc


def _enum_text(value: object) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1] if "." in text else text


def _as_video_profile(profile: Any, description: str) -> Any:
    try:
        video_profile = profile.as_video_stream_profile()
        if not video_profile:
            raise RuntimeError("profile is not a video stream")
        return video_profile
    except Exception as exc:
        raise CameraConfigurationError(
            f"{description} is not a usable video stream profile"
        ) from exc


def _find_video_profile(
    rs: Any,
    pipeline_profile: Any,
    stream_type: Any,
    stream_index: int,
    description: str,
) -> Any:
    matches: list[Any] = []
    try:
        profiles = list(pipeline_profile.get_streams())
    except Exception as exc:
        raise CameraConfigurationError(
            "RealSense did not expose the active stream profiles"
        ) from exc

    for profile in profiles:
        try:
            if profile.stream_type() != stream_type:
                continue
            if stream_index >= 0 and int(profile.stream_index()) != stream_index:
                continue
            matches.append(profile)
        except Exception:
            continue

    if len(matches) != 1:
        available = []
        for profile in profiles:
            try:
                available.append(
                    f"{_enum_text(profile.stream_type())}[{profile.stream_index()}]"
                )
            except Exception:
                available.append("unknown")
        raise CameraConfigurationError(
            f"expected exactly one {description} stream (index {stream_index}); "
            f"active streams are: {', '.join(available) or 'none'}"
        )
    return _as_video_profile(matches[0], description)


def _native_intrinsics(video_profile: Any) -> Any:
    try:
        return video_profile.get_intrinsics()
    except Exception as exc:
        raise CameraConfigurationError(
            "RealSense stream profile did not provide camera intrinsics"
        ) from exc


def _neutral_intrinsics(native: Any) -> CameraIntrinsics:
    try:
        coefficients = tuple(float(value) for value in native.coeffs)
        return CameraIntrinsics(
            width=int(native.width),
            height=int(native.height),
            ppx=float(native.ppx),
            ppy=float(native.ppy),
            fx=float(native.fx),
            fy=float(native.fy),
            distortion_model=_enum_text(native.model),
            coefficients=coefficients,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CameraConfigurationError(
            "RealSense returned malformed camera intrinsics"
        ) from exc


def _stream_diagnostics(name: str, video_profile: Any) -> StreamDiagnostics:
    try:
        return StreamDiagnostics(
            name=name,
            width=int(video_profile.width()),
            height=int(video_profile.height()),
            fps=int(video_profile.fps()),
            pixel_format=_enum_text(video_profile.format()),
            stream_index=int(video_profile.stream_index()),
        )
    except Exception as exc:
        raise CameraConfigurationError(
            f"could not inspect the active {name} stream profile"
        ) from exc


def _safe_device_info(rs: Any, device: Any, field: Any) -> str | None:
    try:
        if hasattr(device, "supports") and not device.supports(field):
            return None
        value = str(device.get_info(field)).strip()
        return value or None
    except Exception:
        return None


def _intrinsics_match(
    first: CameraIntrinsics,
    second: CameraIntrinsics,
    *,
    absolute_tolerance: float = 1e-5,
) -> bool:
    if first.width != second.width or first.height != second.height:
        return False
    if first.distortion_model != second.distortion_model:
        return False
    scalars = (
        (first.ppx, second.ppx),
        (first.ppy, second.ppy),
        (first.fx, second.fx),
        (first.fy, second.fy),
    )
    if not all(
        math.isclose(a, b, rel_tol=1e-7, abs_tol=absolute_tolerance)
        for a, b in scalars
    ):
        return False
    if len(first.coefficients) != len(second.coefficients):
        return False
    return all(
        math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
        for a, b in zip(first.coefficients, second.coefficients)
    )


class _RealSensePipelineSource(FrameSource):
    """Shared native-Depth-viewport implementation for live and bag sources."""

    def __init__(
        self,
        *,
        width: int | None,
        height: int | None,
        fps: int | None,
        warmup_frames: int,
        frame_timeout_ms: int,
        ir_stream_index: int = 1,
        allow_verified_identity_fallback: bool = True,
    ) -> None:
        if width is not None and width <= 0:
            raise CameraConfigurationError("camera width must be positive")
        if height is not None and height <= 0:
            raise CameraConfigurationError("camera height must be positive")
        if fps is not None and fps <= 0:
            raise CameraConfigurationError("camera FPS must be positive")
        if warmup_frames < 0:
            raise CameraConfigurationError("warmup_frames cannot be negative")
        if frame_timeout_ms <= 0:
            raise CameraConfigurationError("frame_timeout_ms must be positive")
        if ir_stream_index <= 0:
            raise CameraConfigurationError(
                "Left IR must use a positive RealSense stream index (normally 1)"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = int(warmup_frames)
        self.frame_timeout_ms = int(frame_timeout_ms)
        self.ir_stream_index = int(ir_stream_index)
        self.allow_verified_identity_fallback = bool(
            allow_verified_identity_fallback
        )

        self._rs: Any | None = None
        self._pipeline: Any | None = None
        self._pipeline_profile: Any | None = None
        self._native_depth_profile: Any | None = None
        self._native_ir_profile: Any | None = None
        self._native_depth_intrinsics: Any | None = None
        self._depth_intrinsics: CameraIntrinsics | None = None
        self._ir_intrinsics: CameraIntrinsics | None = None
        self._depth_scale: float | None = None
        self._aligner: Any | None = None
        self._alignment_strategy: str | None = None
        self._alignment_locked = False
        self._pending_frame: CameraFrame | None = None
        self._started = False
        self._diagnostics: DeviceDiagnostics | None = None

    @property
    def started(self) -> bool:
        return self._started

    @property
    def diagnostics(self) -> DeviceDiagnostics | None:
        """Startup diagnostics, available after a successful ``start``."""

        return self._diagnostics

    def get_diagnostics(self) -> DeviceDiagnostics:
        if self._diagnostics is None:
            raise SourceNotStartedError(
                "source diagnostics are unavailable before start succeeds"
            )
        return self._diagnostics

    def _initialize_running_pipeline(
        self,
        rs: Any,
        pipeline: Any,
        pipeline_profile: Any,
    ) -> None:
        """Resolve calibration, establish alignment, warm up, and prime."""

        self._rs = rs
        self._pipeline = pipeline
        self._pipeline_profile = pipeline_profile
        self._native_depth_profile = _find_video_profile(
            rs,
            pipeline_profile,
            rs.stream.depth,
            0,
            "native Depth",
        )
        self._native_ir_profile = _find_video_profile(
            rs,
            pipeline_profile,
            rs.stream.infrared,
            self.ir_stream_index,
            "Left IR",
        )

        depth_stream = _stream_diagnostics(
            "native Depth", self._native_depth_profile
        )
        ir_stream = _stream_diagnostics("Left IR", self._native_ir_profile)
        if self.width is not None and depth_stream.width != self.width:
            raise CameraConfigurationError(
                f"resolved Depth width {depth_stream.width} != requested {self.width}"
            )
        if self.height is not None and depth_stream.height != self.height:
            raise CameraConfigurationError(
                f"resolved Depth height {depth_stream.height} != requested {self.height}"
            )
        if self.fps is not None and depth_stream.fps != self.fps:
            raise CameraConfigurationError(
                f"resolved Depth FPS {depth_stream.fps} != requested {self.fps}"
            )
        if depth_stream.pixel_format.lower() != "z16":
            raise CameraConfigurationError(
                f"Depth stream must be Z16, got {depth_stream.pixel_format}"
            )
        if ir_stream.pixel_format.lower() != "y8":
            raise CameraConfigurationError(
                f"Left IR stream must be Y8, got {ir_stream.pixel_format}"
            )

        self._native_depth_intrinsics = _native_intrinsics(
            self._native_depth_profile
        )
        self._depth_intrinsics = _neutral_intrinsics(
            self._native_depth_intrinsics
        )
        self._ir_intrinsics = _neutral_intrinsics(
            _native_intrinsics(self._native_ir_profile)
        )

        try:
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
        except Exception as exc:
            raise CameraConfigurationError(
                "selected source did not provide a Depth scale"
            ) from exc
        if not math.isfinite(depth_scale) or depth_scale <= 0.0:
            raise CameraConfigurationError(
                f"invalid Depth scale returned by RealSense: {depth_scale!r}"
            )
        self._depth_scale = depth_scale

        try:
            # Depth is deliberately the target: the Z16 image remains native
            # and Left IR is resampled into its viewport for QR detection.
            self._aligner = rs.align(rs.stream.depth)
        except Exception as exc:
            self._activate_verified_identity_fallback(
                f"rs.align(rs.stream.depth) could not be created: {exc}"
            )

        self._started = True
        for _ in range(self.warmup_frames):
            self._acquire_frame(self.frame_timeout_ms)

        # Prime one frame so start() proves the alignment contract and callers
        # can immediately query a fully validated source.  It is returned by the
        # first get_frames() call rather than silently discarded.
        self._pending_frame = self._acquire_frame(self.frame_timeout_ms)

    def _activate_verified_identity_fallback(self, reason: str) -> None:
        if self._alignment_locked:
            raise FrameAlignmentError(
                "RealSense alignment failed after it had already succeeded; "
                f"refusing to change pixel-coordinate contracts: {reason}"
            )
        if not self.allow_verified_identity_fallback:
            raise FrameAlignmentError(
                "RealSense cannot align Left IR into the native Depth viewport "
                f"and calibrated passthrough is disabled: {reason}"
            )
        if (
            self._native_depth_profile is None
            or self._native_ir_profile is None
            or self._depth_intrinsics is None
            or self._ir_intrinsics is None
        ):
            raise FrameAlignmentError(
                "cannot validate IR/Depth passthrough before profiles are resolved"
            )
        if not _intrinsics_match(self._depth_intrinsics, self._ir_intrinsics):
            raise FrameAlignmentError(
                "RealSense cannot align Left IR into Depth and direct passthrough "
                "is unsafe because the native IR/Depth intrinsics differ. "
                f"Alignment failure: {reason}"
            )

        try:
            extrinsics = self._native_depth_profile.get_extrinsics_to(
                self._native_ir_profile
            )
            rotation = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(
                3, 3
            )
            translation = np.asarray(
                extrinsics.translation, dtype=np.float64
            ).reshape(3)
        except Exception as exc:
            raise FrameAlignmentError(
                "RealSense cannot align Left IR into Depth and the native "
                "IR/Depth extrinsics could not be verified"
            ) from exc
        if not np.allclose(rotation, np.eye(3), rtol=0.0, atol=1e-7):
            raise FrameAlignmentError(
                "RealSense cannot align Left IR into Depth and direct passthrough "
                "is unsafe because IR/Depth rotation is not identity"
            )
        if not np.allclose(translation, np.zeros(3), rtol=0.0, atol=1e-7):
            raise FrameAlignmentError(
                "RealSense cannot align Left IR into Depth and direct passthrough "
                "is unsafe because IR/Depth translation is not zero"
            )

        self._aligner = None
        self._alignment_strategy = (
            "verified native passthrough: Left IR1 and Depth have equal "
            "intrinsics plus identity extrinsics"
        )
        self._alignment_locked = True
        LOGGER.warning(
            "Falling back to strictly verified native IR/Depth correspondence: %s",
            reason,
        )

    def _mapped_frames(self, frameset: Any) -> tuple[Any, Any]:
        if self._aligner is None:
            if not self._alignment_locked:
                self._activate_verified_identity_fallback(
                    "alignment processor is unavailable"
                )
            mapped_frameset = frameset
        else:
            try:
                mapped_frameset = self._aligner.process(frameset)
            except Exception as exc:
                if not self._alignment_locked:
                    self._activate_verified_identity_fallback(str(exc))
                    mapped_frameset = frameset
                else:
                    raise FrameAlignmentError(
                        "failed to align Left IR into the native Depth viewport"
                    ) from exc

        try:
            depth_frame = mapped_frameset.get_depth_frame()
            ir_frame = mapped_frameset.get_infrared_frame(self.ir_stream_index)
        except Exception as exc:
            if self._aligner is not None and not self._alignment_locked:
                self._activate_verified_identity_fallback(
                    "aligned frameset does not expose Left IR stream index 1"
                )
                return self._mapped_frames(frameset)
            raise FrameAlignmentError(
                "frameset does not expose native Depth and aligned Left IR1"
            ) from exc

        if not depth_frame or not ir_frame:
            if self._aligner is not None and not self._alignment_locked:
                self._activate_verified_identity_fallback(
                    "aligned frameset returned an empty Depth or Left IR1 frame"
                )
                return self._mapped_frames(frameset)
            raise FrameAlignmentError(
                "frameset returned an empty native Depth or aligned Left IR1 frame"
            )

        if self._aligner is not None:
            # Aligning *to* Depth must never alter the native Depth viewport.
            try:
                returned_depth = _neutral_intrinsics(
                    depth_frame.profile.as_video_stream_profile().get_intrinsics()
                )
            except Exception as exc:
                raise FrameAlignmentError(
                    "aligned frameset did not preserve a valid Depth profile"
                ) from exc
            if self._depth_intrinsics is None or not _intrinsics_match(
                returned_depth, self._depth_intrinsics
            ):
                raise FrameAlignmentError(
                    "rs.align(rs.stream.depth) changed the native Depth viewport; "
                    "refusing to mix its pixels with native Depth intrinsics"
                )
            try:
                returned_ir_index = int(ir_frame.profile.stream_index())
            except Exception as exc:
                raise FrameAlignmentError(
                    "aligned infrared frame has no verifiable stream index"
                ) from exc
            if returned_ir_index != self.ir_stream_index:
                raise FrameAlignmentError(
                    "aligned infrared frame is not Left IR stream index "
                    f"{self.ir_stream_index} (got {returned_ir_index})"
                )
            self._alignment_strategy = (
                "rs.align(rs.stream.depth): Left IR1 resampled into native "
                "Depth viewport; Z16 Depth remains native"
            )
            self._alignment_locked = True

        return depth_frame, ir_frame

    def _wait_for_frameset(self, timeout_ms: int) -> Any:
        if self._pipeline is None:
            raise SourceNotStartedError("frame source has not been started")
        try:
            return self._pipeline.wait_for_frames(timeout_ms)
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "timeout" in lowered or "didn't arrive" in lowered:
                raise FrameTimeoutError(
                    f"no synchronized RealSense frameset arrived within "
                    f"{timeout_ms} ms"
                ) from exc
            raise FrameAcquisitionError(
                f"RealSense failed while waiting for a frameset: {message}"
            ) from exc

    def _acquire_frame(self, timeout_ms: int) -> CameraFrame:
        frameset = self._wait_for_frameset(timeout_ms)
        depth_frame, ir_frame = self._mapped_frames(frameset)
        try:
            depth_image = np.asanyarray(depth_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())
        except Exception as exc:
            raise FrameAcquisitionError(
                "failed to expose RealSense frame buffers as NumPy arrays"
            ) from exc

        if depth_image.ndim != 2 or ir_image.ndim != 2:
            raise FrameAcquisitionError(
                "expected two-dimensional Z16 Depth and Y8 Left IR images"
            )
        if depth_image.shape != ir_image.shape:
            raise FrameShapeMismatchError(
                "Left IR alignment did not produce the native Depth viewport: "
                f"IR {ir_image.shape}, Depth {depth_image.shape}"
            )
        if depth_image.dtype != np.uint16:
            raise FrameAcquisitionError(
                f"expected native Z16 Depth data, got {depth_image.dtype}"
            )
        if ir_image.dtype != np.uint8:
            raise FrameAcquisitionError(
                f"expected aligned Y8 Left IR data, got {ir_image.dtype}"
            )

        try:
            timestamp_ms = float(depth_frame.get_timestamp())
            frame_number = int(depth_frame.get_frame_number())
        except Exception as exc:
            raise FrameAcquisitionError(
                "RealSense frame is missing timestamp or frame-number metadata"
            ) from exc

        return CameraFrame(
            ir_image=ir_image,
            depth_image=depth_image,
            timestamp_ms=timestamp_ms,
            frame_number=frame_number,
            depth_frame=depth_frame,
            ir_frame=ir_frame,
        )

    def get_frames(self, timeout_ms: int | None = None) -> CameraFrame:
        if not self._started:
            raise SourceNotStartedError("frame source has not been started")
        resolved_timeout = self.frame_timeout_ms if timeout_ms is None else timeout_ms
        if resolved_timeout <= 0:
            raise CameraConfigurationError("frame timeout must be positive")
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return frame
        return self._acquire_frame(int(resolved_timeout))

    def get_depth_intrinsics(self) -> CameraIntrinsics:
        if not self._started or self._depth_intrinsics is None:
            raise SourceNotStartedError(
                "Depth intrinsics are unavailable before the source starts"
            )
        return self._depth_intrinsics

    def get_depth_scale(self) -> float:
        if not self._started or self._depth_scale is None:
            raise SourceNotStartedError(
                "Depth scale is unavailable before the source starts"
            )
        return self._depth_scale

    def deproject_pixel(
        self,
        pixel: tuple[float, float] | list[float] | np.ndarray,
        depth_m: float,
    ) -> np.ndarray:
        if (
            not self._started
            or self._rs is None
            or self._native_depth_intrinsics is None
            or self._depth_intrinsics is None
        ):
            raise SourceNotStartedError(
                "pixel deprojection is unavailable before the source starts"
            )
        pixel_array = np.asarray(pixel, dtype=np.float64)
        if pixel_array.shape != (2,) or not np.all(np.isfinite(pixel_array)):
            raise ValueError("pixel must contain exactly two finite coordinates")
        if not math.isfinite(depth_m) or depth_m <= 0.0:
            raise ValueError("depth_m must be a positive finite value in metres")
        u, v = float(pixel_array[0]), float(pixel_array[1])
        intr = self._depth_intrinsics
        if not (0.0 <= u <= intr.width - 1 and 0.0 <= v <= intr.height - 1):
            raise ValueError(
                f"pixel {(u, v)} lies outside the native Depth viewport "
                f"{intr.width}x{intr.height}"
            )
        try:
            point = self._rs.rs2_deproject_pixel_to_point(
                self._native_depth_intrinsics,
                [u, v],
                float(depth_m),
            )
        except Exception as exc:
            raise FrameAcquisitionError(
                f"RealSense failed to deproject pixel {(u, v)}"
            ) from exc
        point_array = np.asarray(point, dtype=np.float64)
        if point_array.shape != (3,) or not np.all(np.isfinite(point_array)):
            raise FrameAcquisitionError(
                "RealSense returned an invalid deprojected 3D point"
            )
        return point_array

    def _build_diagnostics(
        self,
        *,
        source_type: str,
        camera_model: str,
        serial_number: str,
        firmware_version: str,
        usb_type: str | None,
        emitter_enabled: bool | None,
        recording_path: str | None = None,
        playback_path: str | None = None,
    ) -> DeviceDiagnostics:
        if (
            self._native_depth_profile is None
            or self._native_ir_profile is None
            or self._depth_scale is None
            or self._depth_intrinsics is None
            or self._ir_intrinsics is None
            or self._alignment_strategy is None
        ):
            raise CameraConfigurationError(
                "cannot build diagnostics before calibration and alignment succeed"
            )
        return DeviceDiagnostics(
            source_type=source_type,
            camera_model=camera_model,
            serial_number=serial_number,
            firmware_version=firmware_version,
            usb_type=usb_type,
            depth_stream=_stream_diagnostics(
                "native Depth", self._native_depth_profile
            ),
            left_ir_stream=_stream_diagnostics(
                "Left IR (before alignment)", self._native_ir_profile
            ),
            depth_scale_m=self._depth_scale,
            depth_intrinsics=self._depth_intrinsics,
            left_ir_intrinsics=self._ir_intrinsics,
            emitter_enabled=emitter_enabled,
            alignment_strategy=self._alignment_strategy,
            recording_path=recording_path,
            playback_path=playback_path,
        )

    def stop(self) -> None:
        pipeline = self._pipeline
        self._started = False
        self._pending_frame = None
        self._aligner = None
        self._alignment_locked = False
        self._alignment_strategy = None
        self._pipeline = None
        self._pipeline_profile = None
        self._native_depth_profile = None
        self._native_ir_profile = None
        self._native_depth_intrinsics = None
        self._depth_intrinsics = None
        self._ir_intrinsics = None
        self._depth_scale = None
        self._rs = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                LOGGER.warning("RealSense pipeline stop failed: %s", exc)


class RealSenseCamera(_RealSensePipelineSource):
    """Synchronized D435i Left IR + native Depth live source."""

    def __init__(
        self,
        *,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        emitter_enabled: bool = False,
        warmup_frames: int = 30,
        serial_number: str | None = None,
        record_bag_path: str | Path | None = None,
        frame_timeout_ms: int = 5000,
        validate_d435i: bool = True,
        allow_verified_identity_fallback: bool = True,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            fps=fps,
            warmup_frames=warmup_frames,
            frame_timeout_ms=frame_timeout_ms,
            ir_stream_index=1,
            allow_verified_identity_fallback=allow_verified_identity_fallback,
        )
        self.emitter_enabled = bool(emitter_enabled)
        self.serial_number = (
            serial_number.strip() if isinstance(serial_number, str) else None
        )
        if serial_number is not None and not self.serial_number:
            raise CameraConfigurationError("serial_number cannot be blank")
        self.record_bag_path = (
            Path(record_bag_path).expanduser()
            if record_bag_path is not None
            else None
        )
        self.validate_d435i = bool(validate_d435i)

    def _validate_record_path(self) -> str | None:
        if self.record_bag_path is None:
            return None
        path = self.record_bag_path
        if path.suffix.lower() != ".bag":
            raise CameraConfigurationError(
                f"RealSense recording path must end in .bag: {path}"
            )
        if path.exists():
            raise CameraConfigurationError(
                f"refusing to overwrite existing RealSense bag: {path}"
            )
        if not path.parent.exists() or not path.parent.is_dir():
            raise CameraConfigurationError(
                f"recording directory does not exist: {path.parent}"
            )
        return str(path.resolve())

    def start(self) -> None:
        if self._started:
            return
        recording_path = self._validate_record_path()
        self._diagnostics = None
        rs = _require_realsense()
        pipeline = rs.pipeline()
        config = rs.config()
        try:
            if self.serial_number is not None:
                config.enable_device(self.serial_number)
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )
            config.enable_stream(
                rs.stream.infrared,
                1,
                self.width,
                self.height,
                rs.format.y8,
                self.fps,
            )
            if recording_path is not None:
                config.enable_record_to_file(recording_path)
            pipeline_profile = pipeline.start(config)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise CameraConnectionError(
                "failed to start D435i with native Depth and Left IR1 at "
                f"{self.width}x{self.height}@{self.fps}: {exc}"
            ) from exc

        try:
            device = pipeline_profile.get_device()
            model = _safe_device_info(rs, device, rs.camera_info.name) or "unknown"
            serial = (
                _safe_device_info(rs, device, rs.camera_info.serial_number)
                or "unknown"
            )
            firmware = (
                _safe_device_info(rs, device, rs.camera_info.firmware_version)
                or "unknown"
            )
            usb_type = _safe_device_info(
                rs, device, rs.camera_info.usb_type_descriptor
            )
            if self.validate_d435i and "D435I" not in model.upper():
                raise UnsupportedDeviceError(
                    f"selected device is {model!r}; this source requires a D435i "
                    "(set validate_d435i=False only for a deliberately compatible "
                    "RealSense device)"
                )
            if self.serial_number is not None and serial != self.serial_number:
                raise CameraConnectionError(
                    f"opened serial {serial!r}, expected {self.serial_number!r}"
                )

            try:
                depth_sensor = device.first_depth_sensor()
            except Exception as exc:
                raise CameraConfigurationError(
                    f"selected RealSense device {model!r} has no Depth sensor"
                ) from exc
            emitter_state: bool | None
            try:
                supports_emitter = bool(
                    depth_sensor.supports(rs.option.emitter_enabled)
                )
            except Exception:
                supports_emitter = False
            if supports_emitter:
                try:
                    depth_sensor.set_option(
                        rs.option.emitter_enabled,
                        1.0 if self.emitter_enabled else 0.0,
                    )
                    emitter_state = bool(
                        round(
                            float(
                                depth_sensor.get_option(rs.option.emitter_enabled)
                            )
                        )
                    )
                except Exception as exc:
                    raise CameraConfigurationError(
                        f"failed to set D435i emitter to {self.emitter_enabled}"
                    ) from exc
                if emitter_state != self.emitter_enabled:
                    raise CameraConfigurationError(
                        "D435i did not apply the requested emitter state: "
                        f"requested={self.emitter_enabled}, actual={emitter_state}"
                    )
            elif self.emitter_enabled:
                raise CameraConfigurationError(
                    "emitter_enabled=true was requested but the selected Depth "
                    "sensor does not expose the emitter option"
                )
            else:
                emitter_state = None

            self._initialize_running_pipeline(rs, pipeline, pipeline_profile)
            self._diagnostics = self._build_diagnostics(
                source_type="live_d435i",
                camera_model=model,
                serial_number=serial,
                firmware_version=firmware,
                usb_type=usb_type,
                emitter_enabled=emitter_state,
                recording_path=recording_path,
            )
            LOGGER.info("RealSense source started\n%s", self._diagnostics)
        except Exception as exc:
            # _initialize_running_pipeline may or may not have set self._pipeline.
            self._pipeline = pipeline
            self.stop()
            if isinstance(exc, CameraError):
                raise
            raise CameraConnectionError(
                f"D435i initialization failed after opening the device: {exc}"
            ) from exc
