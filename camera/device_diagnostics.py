"""Structured startup diagnostics for live and recorded RealSense sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .frame_source import CameraIntrinsics


@dataclass(frozen=True, slots=True)
class StreamDiagnostics:
    """A resolved RealSense video stream profile."""

    name: str
    width: int
    height: int
    fps: int
    pixel_format: str
    stream_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "pixel_format": self.pixel_format,
            "stream_index": self.stream_index,
        }


@dataclass(frozen=True, slots=True)
class DeviceDiagnostics:
    """Calibration and provenance captured when a source starts."""

    source_type: str
    camera_model: str
    serial_number: str
    firmware_version: str
    usb_type: str | None
    depth_stream: StreamDiagnostics
    left_ir_stream: StreamDiagnostics
    depth_scale_m: float
    depth_intrinsics: CameraIntrinsics
    left_ir_intrinsics: CameraIntrinsics
    emitter_enabled: bool | None
    alignment_strategy: str
    coordinate_system: str = "native_depth_viewport"
    recording_path: str | None = None
    playback_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "source_type": self.source_type,
            "camera_model": self.camera_model,
            "serial_number": self.serial_number,
            "firmware_version": self.firmware_version,
            "usb_type": self.usb_type,
            "depth_stream": self.depth_stream.as_dict(),
            "left_ir_stream": self.left_ir_stream.as_dict(),
            "depth_scale_m": self.depth_scale_m,
            "depth_intrinsics": self.depth_intrinsics.as_dict(),
            "left_ir_intrinsics": self.left_ir_intrinsics.as_dict(),
            "emitter_enabled": self.emitter_enabled,
            "alignment_strategy": self.alignment_strategy,
            "coordinate_system": self.coordinate_system,
            "recording_path": self.recording_path,
            "playback_path": self.playback_path,
        }

    def format_text(self) -> str:
        """Format all required startup facts for logs and experiment records."""

        depth = self.depth_stream
        infrared = self.left_ir_stream
        intr = self.depth_intrinsics
        emitter = (
            "unavailable/recorded"
            if self.emitter_enabled is None
            else ("ON" if self.emitter_enabled else "OFF")
        )
        lines = [
            f"Source: {self.source_type}",
            f"Camera model: {self.camera_model}",
            f"Serial number: {self.serial_number}",
            f"Firmware version: {self.firmware_version}",
        ]
        if self.usb_type:
            lines.append(f"USB type: {self.usb_type}")
        lines.extend(
            [
                (
                    "Depth stream: "
                    f"{depth.width}x{depth.height} @ {depth.fps} FPS "
                    f"{depth.pixel_format} (index {depth.stream_index})"
                ),
                (
                    "Left IR stream: "
                    f"{infrared.width}x{infrared.height} @ {infrared.fps} FPS "
                    f"{infrared.pixel_format} (index {infrared.stream_index})"
                ),
                f"Depth scale: {self.depth_scale_m:.12g} m/unit",
                (
                    "Depth intrinsics: "
                    f"fx={intr.fx:.9g}, fy={intr.fy:.9g}, "
                    f"ppx={intr.ppx:.9g}, ppy={intr.ppy:.9g}"
                ),
                f"Depth distortion model: {intr.distortion_model}",
                f"Depth distortion coefficients: {list(intr.coefficients)}",
                f"Emitter: {emitter}",
                f"Pixel alignment: {self.alignment_strategy}",
                f"Returned coordinate system: {self.coordinate_system}",
            ]
        )
        if self.recording_path:
            lines.append(f"Recording bag: {self.recording_path}")
        if self.playback_path:
            lines.append(f"Playback bag: {self.playback_path}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_text()
