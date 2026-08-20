from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import numpy as np

from camera import (
    BagPlaybackError,
    CameraFrame,
    FrameShapeMismatchError,
    RealSenseBagSource,
    RealSenseCamera,
    RealSenseUnavailableError,
)
from camera.realsense_camera import _RealSensePipelineSource


def _frame(sequence: int) -> CameraFrame:
    return CameraFrame(
        ir_image=np.zeros((2, 3), dtype=np.uint8),
        depth_image=np.zeros((2, 3), dtype=np.uint16),
        timestamp_ms=float(sequence),
        frame_number=sequence,
    )


class CameraContractTests(unittest.TestCase):
    def test_camera_frame_enforces_shared_depth_viewport(self) -> None:
        with self.assertRaises(FrameShapeMismatchError):
            CameraFrame(
                ir_image=np.zeros((2, 3), dtype=np.uint8),
                depth_image=np.zeros((3, 2), dtype=np.uint16),
                timestamp_ms=0.0,
                frame_number=1,
            )

    def test_live_source_import_is_lazy_without_sdk(self) -> None:
        source = RealSenseCamera(warmup_frames=0)
        with patch.dict(sys.modules, {"pyrealsense2": None}):
            with self.assertRaises(RealSenseUnavailableError):
                source.start()

    def test_repeating_bag_skips_warmup_after_each_loop_restart(self) -> None:
        source = RealSenseBagSource(
            "synthetic.bag",
            repeat_playback=True,
            warmup_frames=2,
        )
        # Frame 100 represents the already-warmed first pass.  Timestamp/frame
        # rollback marks the next frame 0 as the start of a repeated pass.
        frames = [_frame(100), _frame(0), _frame(1), _frame(2)]
        with patch.object(
            _RealSensePipelineSource,
            "get_frames",
            side_effect=frames,
        ) as base_get_frames:
            self.assertEqual(source.get_frames().frame_number, 100)
            self.assertEqual(source.get_frames().frame_number, 2)

        # The second public result consumed two warmup frames first.
        self.assertEqual(base_get_frames.call_count, 4)

    def test_repeating_bag_short_loop_raises_instead_of_spinning(self) -> None:
        for loop_frames in (1, 2):
            with self.subTest(loop_frames=loop_frames):
                source = RealSenseBagSource(
                    "synthetic.bag",
                    repeat_playback=True,
                    warmup_frames=2,
                )
                repeated = [_frame(index) for index in range(loop_frames)]
                frames = [_frame(100), *repeated, _frame(0)]
                with patch.object(
                    _RealSensePipelineSource,
                    "get_frames",
                    side_effect=frames,
                ):
                    self.assertEqual(source.get_frames().frame_number, 100)
                    with self.assertRaisesRegex(
                        BagPlaybackError,
                        "no frame after camera.warmup_frames",
                    ):
                        source.get_frames()


if __name__ == "__main__":
    unittest.main()
