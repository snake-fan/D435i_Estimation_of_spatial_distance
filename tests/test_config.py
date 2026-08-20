from __future__ import annotations

import unittest
from dataclasses import replace

from utils.config import (
    AppConfig,
    CameraConfig,
    ConfigError,
    PlaneConfig,
    QRConfig,
    TemporalConfig,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        validate_config(AppConfig())

    def test_expected_ids_must_be_two_distinct_payloads(self) -> None:
        config = replace(AppConfig(), qr=QRConfig(expected_ids=["QR_A", "QR_A"]))
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_temporal_minimum_cannot_exceed_window(self) -> None:
        config = replace(
            AppConfig(), temporal=TemporalConfig(window_size=5, min_valid_frames=6)
        )
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_string_boolean_is_rejected_instead_of_becoming_true(self) -> None:
        config = replace(
            AppConfig(), camera=replace(CameraConfig(), emitter_enabled="false")
        )
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_non_finite_plane_threshold_is_rejected(self) -> None:
        config = replace(
            AppConfig(), plane=replace(PlaneConfig(), distance_threshold=float("inf"))
        )
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_rms_gate_must_be_below_ransac_inlier_threshold(self) -> None:
        config = replace(
            AppConfig(),
            plane=replace(
                PlaneConfig(), distance_threshold=0.004, max_rms=0.006
            ),
        )
        with self.assertRaises(ConfigError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
