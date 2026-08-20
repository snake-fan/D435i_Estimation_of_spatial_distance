"""Finite RealSense ``.bag`` playback using the same frame contract as live."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Event
from typing import Any

from .frame_source import (
    BagEndOfStream,
    BagPlaybackError,
    CameraConfigurationError,
    CameraError,
    CameraFrame,
    FrameAcquisitionError,
)
from .realsense_camera import (
    _RealSensePipelineSource,
    _require_realsense,
    _safe_device_info,
)

LOGGER = logging.getLogger(__name__)


class RealSenseBagSource(_RealSensePipelineSource):
    """Play a D435i bag without changing any upper-layer measurement code.

    By default playback is non-realtime, so algorithm throughput does not drop
    recorded frames.  A non-repeating source raises :class:`BagEndOfStream` at
    normal EOF and exposes the same state through :attr:`is_eof`.
    """

    def __init__(
        self,
        bag_path: str | Path,
        *,
        repeat_playback: bool = False,
        real_time: bool = False,
        warmup_frames: int = 0,
        frame_timeout_ms: int = 2000,
        validate_d435i: bool = True,
        allow_verified_identity_fallback: bool = True,
    ) -> None:
        super().__init__(
            width=None,
            height=None,
            fps=None,
            warmup_frames=warmup_frames,
            frame_timeout_ms=frame_timeout_ms,
            ir_stream_index=1,
            allow_verified_identity_fallback=allow_verified_identity_fallback,
        )
        self.bag_path = Path(bag_path).expanduser()
        self.repeat_playback = bool(repeat_playback)
        self.real_time = bool(real_time)
        self.validate_d435i = bool(validate_d435i)
        self._playback: Any | None = None
        self._status_callback: Any | None = None
        self._stopped_event = Event()
        self._frames_received = 0
        self._last_delivered_timestamp_ms: float | None = None
        self._last_delivered_frame_number: int | None = None
        self._loop_warmup_remaining = 0
        self._awaiting_post_warmup_frame = False

    def _resolved_bag_path(self) -> str:
        path = self.bag_path
        if path.suffix.lower() != ".bag":
            raise CameraConfigurationError(
                f"RealSense playback path must end in .bag: {path}"
            )
        if not path.exists():
            raise CameraConfigurationError(f"RealSense bag does not exist: {path}")
        if not path.is_file():
            raise CameraConfigurationError(
                f"RealSense bag path is not a regular file: {path}"
            )
        return str(path.resolve())

    def _handle_playback_status(self, status: Any) -> None:
        """SDK callback; it may execute on a RealSense worker thread."""

        rs = self._rs
        if rs is None or self.repeat_playback:
            return
        try:
            if status == rs.playback_status.stopped:
                self._stopped_event.set()
            elif status == rs.playback_status.playing:
                self._stopped_event.clear()
        except Exception:
            # A status callback must never raise back into the SDK worker.
            LOGGER.debug("Could not interpret RealSense playback status %r", status)

    def _playback_stopped(self) -> bool:
        if self.repeat_playback or self._playback is None or self._rs is None:
            return False
        try:
            stopped = (
                self._playback.current_status()
                == self._rs.playback_status.stopped
            )
        except Exception:
            stopped = self._stopped_event.is_set()
        if stopped:
            self._stopped_event.set()
        return stopped

    @property
    def is_eof(self) -> bool:
        return (
            self._started
            and self._pending_frame is None
            and self._frames_received > 0
            and self._playback_stopped()
        )

    def _wait_for_frameset(self, timeout_ms: int) -> Any:
        if self._playback_stopped() and self._frames_received > 0:
            # Playback may become "stopped" with a final frameset still queued.
            # Drain it before reporting EOF so a finite bag is not truncated.
            try:
                queued = self._pipeline.poll_for_frames() if self._pipeline else None
            except Exception:
                queued = None
            if queued:
                self._frames_received += 1
                return queued
            raise BagEndOfStream(
                f"end of RealSense bag reached: {self.bag_path}"
            )

        try:
            frameset = super()._wait_for_frameset(timeout_ms)
        except FrameAcquisitionError as exc:
            if self._playback_stopped() and self._frames_received > 0:
                raise BagEndOfStream(
                    f"end of RealSense bag reached: {self.bag_path}"
                ) from exc
            raise
        self._frames_received += 1
        return frameset

    def start(self) -> None:
        if self._started:
            return
        bag_path = self._resolved_bag_path()
        self._diagnostics = None
        rs = _require_realsense()
        self._stopped_event.clear()
        self._frames_received = 0
        pipeline = rs.pipeline()
        config = rs.config()
        try:
            config.enable_device_from_file(bag_path, self.repeat_playback)
            pipeline_profile = pipeline.start(config)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise BagPlaybackError(
                f"failed to open RealSense bag {bag_path}: {exc}"
            ) from exc

        try:
            device = pipeline_profile.get_device()
            try:
                playback = device.as_playback()
                if not playback:
                    raise RuntimeError("device is not a playback device")
            except Exception as exc:
                raise BagPlaybackError(
                    f"file did not create a RealSense playback device: {bag_path}"
                ) from exc

            self._rs = rs
            self._playback = playback
            playback.set_real_time(self.real_time)
            self._status_callback = self._handle_playback_status
            try:
                playback.set_status_changed_callback(self._status_callback)
            except Exception as exc:
                raise BagPlaybackError(
                    "RealSense playback does not support an EOF status callback"
                ) from exc

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
                raise BagPlaybackError(
                    f"bag was recorded by {model!r}, not a D435i; set "
                    "validate_d435i=False only for a deliberately compatible bag"
                )

            emitter_state: bool | None = None
            try:
                depth_sensor = device.first_depth_sensor()
                if depth_sensor.supports(rs.option.emitter_enabled):
                    emitter_state = bool(
                        round(
                            float(
                                depth_sensor.get_option(rs.option.emitter_enabled)
                            )
                        )
                    )
            except Exception:
                # Some bags do not retain a readable emitter option.
                emitter_state = None

            self._initialize_running_pipeline(rs, pipeline, pipeline_profile)
            self._diagnostics = self._build_diagnostics(
                source_type="bag_playback",
                camera_model=model,
                serial_number=serial,
                firmware_version=firmware,
                usb_type=usb_type,
                emitter_enabled=emitter_state,
                playback_path=bag_path,
            )
            LOGGER.info("RealSense bag source started\n%s", self._diagnostics)
        except Exception as exc:
            self._pipeline = pipeline
            self.stop()
            if isinstance(exc, CameraError):
                raise
            raise BagPlaybackError(
                f"RealSense bag initialization failed for {bag_path}: {exc}"
            ) from exc

    def _consume_loop_warmup(self, frame: CameraFrame) -> bool:
        """Return whether ``frame`` belongs to a repeated loop's warmup head."""

        same_recorded_frame = (
            self._last_delivered_timestamp_ms is not None
            and self._last_delivered_frame_number is not None
            and frame.timestamp_ms == self._last_delivered_timestamp_ms
            and frame.frame_number == self._last_delivered_frame_number
        )
        restarted = (
            self.repeat_playback
            and self._last_delivered_timestamp_ms is not None
            and (
                frame.timestamp_ms < self._last_delivered_timestamp_ms
                or (
                    self._last_delivered_frame_number is not None
                    and frame.frame_number < self._last_delivered_frame_number
                )
                # A one-frame recording restarts at exactly the same SDK
                # timestamp/frame number, so there is no numerical rollback.
                or same_recorded_frame
            )
        )
        if restarted:
            if self._awaiting_post_warmup_frame:
                raise BagPlaybackError(
                    "bag loop has no frame after camera.warmup_frames; "
                    "cannot produce a post-warmup frame"
                )
            self._loop_warmup_remaining = self.warmup_frames
            self._awaiting_post_warmup_frame = self.warmup_frames > 0

        self._last_delivered_timestamp_ms = frame.timestamp_ms
        self._last_delivered_frame_number = frame.frame_number
        if self._loop_warmup_remaining > 0:
            self._loop_warmup_remaining -= 1
            return True
        # Reaching this branch proves the repeated recording contains at least
        # one frame after its warmup head.  Keep the state set while consuming
        # the last warmup frame so a loop of exactly ``warmup_frames`` fails at
        # the next restart instead of spinning forever.
        self._awaiting_post_warmup_frame = False
        return False

    def get_frames(self, timeout_ms: int | None = None) -> CameraFrame:
        while True:
            frame = super().get_frames(timeout_ms)
            if not self._consume_loop_warmup(frame):
                return frame

    def stop(self) -> None:
        super().stop()
        self._playback = None
        self._status_callback = None
        self._stopped_event.clear()
        self._frames_received = 0
        self._last_delivered_timestamp_ms = None
        self._last_delivered_frame_number = None
        self._loop_warmup_remaining = 0
        self._awaiting_post_warmup_frame = False


# Short alias useful for CLI/source-factory code.
BagSource = RealSenseBagSource
