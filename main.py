#!/usr/bin/env python3
"""D435i + two QR codes: real-time 3D Euclidean distance measurement."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from camera import (
    BagEndOfStream,
    CameraError,
    FrameAcquisitionError,
    RealSenseBagSource,
    RealSenseCamera,
)
from detection import QRDetector
from evaluation.baseline_methods import (
    center_single_pixel_point,
    roi_median_depth_point,
)
from geometry.quadrilateral import polygon_mask
from measurement import (
    DistanceMeasure,
    MeasurementStatus,
    QR3DLocator,
    QR3DResult,
    euclidean_distance,
)
from recording import (
    CSVRecorder,
    JsonLinesWriter,
    PointCloudDumper,
    build_csv_row,
    build_json_result,
)
from statistics import TemporalStatistics
from utils import AppConfig, ConfigError, load_config, validate_config


LOGGER = logging.getLogger("d435_qr_measure")
VERSION = "0.1.0"
MAX_CONSECUTIVE_ACQUISITION_ERRORS = 5


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the camera-space Euclidean distance between QR_A and QR_B "
            "with an Intel RealSense D435i."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bag", type=Path, help="Replay a RealSense .bag file")
    source.add_argument(
        "--record-bag",
        type=Path,
        help="Record the live Depth + Left IR streams to a new .bag file",
    )
    parser.add_argument(
        "--serial", help="Override camera.serial_number for live acquisition"
    )
    parser.add_argument(
        "--repeat-bag", action="store_true", help="Loop bag playback instead of exiting at EOF"
    )
    parser.add_argument(
        "--bag-real-time",
        action="store_true",
        help="Honor recorded timing (default processes every frame as fast as possible)",
    )
    parser.add_argument(
        "--allow-compatible-device",
        action="store_true",
        help="Allow a compatible RealSense model instead of enforcing D435i metadata",
    )
    parser.add_argument("--debug", action="store_true", help="Detailed logs and overlay")
    parser.add_argument(
        "--record", type=Path, metavar="CSV", help="Write every frame to a new CSV file"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        metavar="JSONL",
        help="Write JSON Lines to a new file; use '-' for stdout",
    )
    parser.add_argument(
        "--dump-points",
        nargs="?",
        const=Path("debug_points"),
        type=Path,
        metavar="DIR",
        help="Continuously overwrite latest qr_a/qr_b ROI point-cloud CSVs",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable the OpenCV window"
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        help="Stop after this many frames (useful for bag and smoke tests)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def _configure_logging(level: str, debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _validate_cli(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.repeat_bag and args.bag is None:
        parser.error("--repeat-bag requires --bag")
    if args.bag_real_time and args.bag is None:
        parser.error("--bag-real-time requires --bag")
    if args.serial and args.bag is not None:
        parser.error("--serial applies only to a live camera, not --bag")


def _new_timestamped_csv() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return Path(f"measurements_{stamp}.csv")


def _ensure_new_output(path: Path, option: str) -> None:
    if path.exists():
        raise RuntimeError(
            f"{option} refuses to overwrite an existing path: {path}. "
            "Choose a new output name."
        )


def _create_source(args: argparse.Namespace, config: AppConfig) -> Any:
    camera = config.camera
    validate_d435i = not args.allow_compatible_device
    if args.bag is not None:
        return RealSenseBagSource(
            args.bag,
            repeat_playback=args.repeat_bag,
            real_time=args.bag_real_time,
            # App-created bags include the live source's discarded startup
            # frames because SDK recording begins at pipeline.start().  Skip
            # the same count on replay so regression runs start at the same
            # first measurement frame.  Set camera.warmup_frames=0 for bags
            # that were deliberately trimmed elsewhere.
            warmup_frames=camera.warmup_frames,
            frame_timeout_ms=camera.frame_timeout_ms,
            validate_d435i=validate_d435i,
        )
    return RealSenseCamera(
        width=camera.width,
        height=camera.height,
        fps=camera.fps,
        emitter_enabled=camera.emitter_enabled,
        warmup_frames=camera.warmup_frames,
        serial_number=args.serial or camera.serial_number,
        record_bag_path=args.record_bag,
        frame_timeout_ms=camera.frame_timeout_ms,
        validate_d435i=validate_d435i,
    )


def _invalid_result(qr_id: str, reason: str) -> QR3DResult:
    return QR3DResult(
        qr_id=qr_id,
        status=MeasurementStatus.INVALID,
        reject_reasons=(reason,),
    )


def _process_frame(
    *,
    frame: Any,
    expected_ids: Sequence[str],
    detector: QRDetector,
    locator: QR3DLocator,
    intrinsics: Any,
    depth_scale: float,
) -> dict[str, QR3DResult]:
    if frame.ir_image.shape != frame.depth_image.shape:
        return {
            qr_id: _invalid_result(qr_id, "depth_ir_shape_mismatch")
            for qr_id in expected_ids
        }
    try:
        detections = detector.detect(frame.ir_image)
    except Exception as exc:
        LOGGER.warning("QR detection failed for frame %s: %s", frame.frame_number, exc)
        detections = {}

    results: dict[str, QR3DResult] = {}
    for qr_id in expected_ids:
        try:
            results[qr_id] = locator.locate(
                detections.get(qr_id),
                frame.depth_image,
                intrinsics,
                depth_scale,
                qr_id=qr_id,
            )
        except Exception as exc:
            # An individual QR/Depth anomaly invalidates this frame, never the
            # entire acquisition process.  Debug logs retain the traceback.
            LOGGER.warning(
                "3D localization failed for %s at frame %s: %s",
                qr_id,
                frame.frame_number,
                exc,
                exc_info=LOGGER.isEnabledFor(logging.DEBUG),
            )
            results[qr_id] = _invalid_result(qr_id, "processing_error")
    return results


def _baseline_distances(
    *,
    frame: Any,
    qr_results: dict[str, QR3DResult],
    expected_ids: Sequence[str],
    intrinsics: Any,
    depth_scale: float,
    config: AppConfig,
    source: Any,
) -> tuple[float | None, float | None]:
    """Compute research-only Method A/B values for the comparison CSV."""

    def deprojector(_intrinsics: Any, pixel: list[float], depth_m: float) -> Any:
        return source.deproject_pixel(pixel, depth_m)

    method_a_points: list[Any] = []
    method_b_points: list[Any] = []
    for qr_id in expected_ids:
        result = qr_results[qr_id]
        center = result.center_uv
        method_a_points.append(
            None
            if center is None
            else center_single_pixel_point(
                frame.depth_image,
                center,
                intrinsics,
                depth_scale,
                config.measurement.min_depth,
                config.measurement.max_depth,
                deprojector,
            )
        )
        if center is None or result.roi_corners is None:
            method_b_points.append(None)
            continue
        try:
            mask = polygon_mask(frame.depth_image.shape, result.roi_corners)
        except ValueError:
            method_b_points.append(None)
            continue
        method_b_points.append(
            roi_median_depth_point(
                frame.depth_image,
                mask,
                center,
                intrinsics,
                depth_scale,
                config.measurement.min_depth,
                config.measurement.max_depth,
                deprojector,
            )
        )

    def pair_distance(points: list[Any]) -> float | None:
        if len(points) != 2 or points[0] is None or points[1] is None:
            return None
        try:
            return euclidean_distance(points[0], points[1])
        except (TypeError, ValueError):
            return None

    return pair_distance(method_a_points), pair_distance(method_b_points)


def _open_outputs(
    args: argparse.Namespace,
    config: AppConfig,
) -> tuple[CSVRecorder | None, JsonLinesWriter | None, PointCloudDumper | None]:
    csv_path = args.record
    if csv_path is None and config.recording.csv_enabled:
        csv_path = _new_timestamped_csv()
    if csv_path is not None:
        _ensure_new_output(csv_path, "--record")
    if args.json_output is not None and args.json_output != "-":
        _ensure_new_output(Path(args.json_output), "--json-output")

    csv_recorder: CSVRecorder | None = None
    json_writer: JsonLinesWriter | None = None
    created_paths: list[Path] = []
    try:
        if csv_path is not None:
            csv_recorder = CSVRecorder(
                csv_path, flush_every_row=config.recording.flush_every_row
            )
            csv_recorder.open()
            created_paths.append(csv_recorder.path)
            LOGGER.info("CSV recording enabled: %s", csv_path)

        if args.json_output is not None:
            json_writer = JsonLinesWriter(
                args.json_output, flush_every_row=config.recording.flush_every_row
            )
            json_writer.open()
            if args.json_output != "-":
                created_paths.append(Path(args.json_output))

        dumper = PointCloudDumper(args.dump_points) if args.dump_points else None
        return csv_recorder, json_writer, dumper
    except Exception:
        for label, output in (("JSONL", json_writer), ("CSV", csv_recorder)):
            if output is None:
                continue
            try:
                output.close()
            except Exception:
                LOGGER.error(
                    "Could not close partially initialized %s output",
                    label,
                    exc_info=True,
                )
        for path in reversed(created_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.error(
                    "Could not remove partially initialized output %s",
                    path,
                    exc_info=True,
                )
        raise


def run(args: argparse.Namespace, config: AppConfig) -> int:
    # ``run`` is public and can be called without going through ``load_config``.
    # Enforce the same type/range contract for every entry path.
    validate_config(config)

    # Fail before opening USB if the QR dependency is unavailable or broken.
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "OpenCV could not be imported; install a working opencv-python build"
        ) from exc

    source = _create_source(args, config)
    csv_recorder: CSVRecorder | None = None
    json_writer: JsonLinesWriter | None = None
    view: Any | None = None
    processed_frames = 0
    exit_code = 0

    try:
        source.start()
        diagnostics = source.get_diagnostics()
        LOGGER.info("Camera/source diagnostics:\n%s", diagnostics.format_text())
        intrinsics = source.get_depth_intrinsics()
        depth_scale = source.get_depth_scale()

        detector = QRDetector(
            expected_ids=config.qr.expected_ids,
            refine_corners=config.qr.corner_refinement,
        )
        def source_deprojector(
            _intrinsics: Any, pixel: list[float], depth_m: float
        ) -> Any:
            return source.deproject_pixel(pixel, depth_m)

        locator = QR3DLocator.from_config(
            config,
            deprojector=source_deprojector,
        )
        distance_measure = DistanceMeasure()
        temporal = TemporalStatistics(
            window_size=config.temporal.window_size,
            min_valid_frames=config.temporal.min_valid_frames,
        )
        csv_recorder, json_writer, dumper = _open_outputs(args, config)

        display_enabled = (
            not args.no_display
            and (config.visualization.enabled or args.debug)
        )
        if display_enabled:
            from visualization import DebugView

            view = DebugView(
                expected_ids=config.qr.expected_ids,
                depth_scale=depth_scale,
                depth_min_m=config.visualization.depth_min_m,
                depth_max_m=config.visualization.depth_max_m,
                enabled=True,
                detailed=args.debug,
            )

        last_frame_time = time.perf_counter()
        fps_ema: float | None = None
        acquisition_errors = 0

        while True:
            try:
                frame = source.get_frames()
                acquisition_errors = 0
            except BagEndOfStream:
                LOGGER.info("Bag playback reached end of stream")
                temporal.reset()
                break
            except FrameAcquisitionError as exc:
                acquisition_errors += 1
                LOGGER.warning(
                    "Frame acquisition failed (%d/%d): %s",
                    acquisition_errors,
                    MAX_CONSECUTIVE_ACQUISITION_ERRORS,
                    exc,
                )
                # No trustworthy device timestamp exists for a failed
                # acquisition.  Reset immediately rather than mixing a stale
                # pre-timeout window with measurements after recovery.
                temporal.reset()
                if acquisition_errors >= MAX_CONSECUTIVE_ACQUISITION_ERRORS:
                    raise CameraError(
                        "too many consecutive frame acquisition failures"
                    ) from exc
                continue

            processed_frames += 1
            now = time.perf_counter()
            elapsed = now - last_frame_time
            last_frame_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps_ema = instant_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * instant_fps

            qr_results = _process_frame(
                frame=frame,
                expected_ids=config.qr.expected_ids,
                detector=detector,
                locator=locator,
                intrinsics=intrinsics,
                depth_scale=depth_scale,
            )
            first_id, second_id = config.qr.expected_ids
            distance_result = distance_measure.compute(
                qr_results[first_id], qr_results[second_id]
            )
            temporal.update(
                distance_result.distance_m,
                valid=distance_result.valid,
                timestamp_ms=frame.timestamp_ms,
            )
            temporal_snapshot = temporal.get(frame.timestamp_ms)
            status = str(distance_result.status)
            reject_reason = distance_result.reject_reason

            payload = build_json_result(
                source_timestamp_ms=frame.timestamp_ms,
                frame_number=frame.frame_number,
                qr_results=qr_results,
                expected_ids=config.qr.expected_ids,
                distance_result=distance_result,
                temporal=temporal_snapshot,
                status=status,
                reject_reason=reject_reason,
            )
            if json_writer is not None:
                json_writer.write(payload)

            if csv_recorder is not None and csv_recorder.enabled:
                method_a_m, method_b_m = _baseline_distances(
                    frame=frame,
                    qr_results=qr_results,
                    expected_ids=config.qr.expected_ids,
                    intrinsics=intrinsics,
                    depth_scale=depth_scale,
                    config=config,
                    source=source,
                )
                csv_recorder.write(
                    build_csv_row(
                        source_timestamp_ms=frame.timestamp_ms,
                        frame_number=frame.frame_number,
                        qr_results=qr_results,
                        expected_ids=config.qr.expected_ids,
                        distance_result=distance_result,
                        temporal=temporal_snapshot,
                        status=status,
                        reject_reason=reject_reason,
                        method_a_distance_m=method_a_m,
                        method_b_distance_m=method_b_m,
                        method_c_distance_m=distance_result.distance_m,
                    )
                )

            if dumper is not None:
                for qr_id, result in qr_results.items():
                    try:
                        dumper.dump(qr_id, result)
                    except (OSError, ValueError) as exc:
                        LOGGER.warning("Could not dump %s point cloud: %s", qr_id, exc)

            if args.debug:
                LOGGER.debug("Frame result: %s", payload)

            if view is not None:
                from visualization import ViewAction

                action = view.render(
                    frame=frame,
                    qr_results=qr_results,
                    distance_result=distance_result,
                    temporal=temporal_snapshot,
                    status=status,
                    reject_reason=reject_reason,
                    fps=fps_ema,
                    recording=(csv_recorder is not None and csv_recorder.enabled),
                )
                if action is ViewAction.QUIT:
                    break
                if action is ViewAction.TOGGLE_RECORDING:
                    if csv_recorder is None:
                        csv_path = _new_timestamped_csv()
                        csv_recorder = CSVRecorder(
                            csv_path,
                            flush_every_row=config.recording.flush_every_row,
                        )
                        csv_recorder.open()
                        LOGGER.info("CSV recording enabled: %s", csv_path)
                    else:
                        enabled = csv_recorder.toggle()
                        LOGGER.info("CSV recording %s", "enabled" if enabled else "paused")

            if args.max_frames is not None and processed_frames >= args.max_frames:
                LOGGER.info("Reached --max-frames=%d", args.max_frames)
                break

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        exit_code = 130
    finally:
        if csv_recorder is not None:
            try:
                csv_recorder.close()
            except Exception:
                LOGGER.error("Could not close CSV output cleanly", exc_info=True)
        if json_writer is not None:
            try:
                json_writer.close()
            except Exception:
                LOGGER.error("Could not close JSONL output cleanly", exc_info=True)
        if view is not None:
            try:
                view.close()
            except Exception:
                LOGGER.debug("Could not close OpenCV window cleanly", exc_info=True)
        try:
            source.stop()
        except Exception:
            LOGGER.error("Could not stop frame source cleanly", exc_info=True)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_cli(parser, args)
    _configure_logging(args.log_level, args.debug)
    try:
        config = load_config(args.config)
        return run(args, config)
    except ConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2
    except CameraError as exc:
        LOGGER.error("RealSense error: %s", exc)
        return 3
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Application error: %s", exc)
        if args.debug:
            LOGGER.debug("Application failure details", exc_info=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
