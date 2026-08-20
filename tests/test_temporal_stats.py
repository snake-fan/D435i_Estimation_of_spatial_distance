from __future__ import annotations

import unittest

from statistics.temporal_stats import TemporalStatistics


class TemporalStatisticsTests(unittest.TestCase):
    def test_empty_window_is_not_ready_and_contains_no_nan(self) -> None:
        stats = TemporalStatistics(window_size=5, min_valid_frames=3).get()
        self.assertEqual(stats.count, 0)
        self.assertFalse(stats.ready)
        self.assertIsNone(stats.median_m)
        self.assertIsNone(stats.std_m)

    def test_statistics_and_ready_threshold(self) -> None:
        tracker = TemporalStatistics(window_size=5, min_valid_frames=3)
        tracker.add(1.0, 0.0)
        tracker.add(2.0, 10.0)
        self.assertFalse(tracker.get().ready)
        tracker.add(3.0, 20.0)
        stats = tracker.get()
        self.assertTrue(stats.ready)
        self.assertEqual(stats.count, 3)
        self.assertAlmostEqual(stats.median_m or -1.0, 2.0)
        self.assertAlmostEqual(stats.mean_m or -1.0, 2.0)
        self.assertAlmostEqual(stats.mad_m or -1.0, 1.0)
        self.assertAlmostEqual(stats.std_m or -1.0, (2.0 / 3.0) ** 0.5)

    def test_window_keeps_only_most_recent_values(self) -> None:
        tracker = TemporalStatistics(window_size=3, min_valid_frames=1)
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            tracker.add(value, float(index))
        stats = tracker.get()
        self.assertEqual(stats.count, 3)
        self.assertAlmostEqual(stats.median_m or -1.0, 3.0)

    def test_nonfinite_negative_and_explicit_invalid_never_enter_window(self) -> None:
        tracker = TemporalStatistics(
            window_size=5,
            min_valid_frames=1,
            max_invalid_streak=10,
        )
        self.assertFalse(tracker.add(float("nan"), 0.0))
        self.assertFalse(tracker.add(float("inf"), 1.0))
        self.assertFalse(tracker.add(-0.1, 2.0))
        self.assertFalse(tracker.add(1.0, 3.0, valid=False))
        self.assertEqual(tracker.get().count, 0)
        self.assertEqual(tracker.get().invalid_streak, 4)

    def test_invalid_streak_clears_and_marks_old_window_stale(self) -> None:
        tracker = TemporalStatistics(
            window_size=5,
            min_valid_frames=1,
            max_invalid_streak=2,
            stale_after_ms=1_000.0,
        )
        tracker.add(0.5, 0.0)
        tracker.add_invalid(10.0)
        self.assertEqual(tracker.get().count, 1)
        tracker.add_invalid(20.0)
        stats = tracker.get()
        self.assertEqual(stats.count, 0)
        self.assertTrue(stats.stale)
        self.assertFalse(stats.ready)

    def test_elapsed_time_expires_old_window_and_new_value_starts_fresh(self) -> None:
        tracker = TemporalStatistics(
            window_size=5,
            min_valid_frames=2,
            stale_after_ms=100.0,
        )
        tracker.add(0.5, 0.0)
        expired = tracker.get(101.0)
        self.assertEqual(expired.count, 0)
        self.assertTrue(expired.stale)
        tracker.add(0.7, 102.0)
        fresh = tracker.get()
        self.assertEqual(fresh.count, 1)
        self.assertFalse(fresh.stale)
        self.assertAlmostEqual(fresh.median_m or -1.0, 0.7)

    def test_backwards_timestamp_starts_new_epoch(self) -> None:
        tracker = TemporalStatistics(window_size=5, min_valid_frames=1)
        tracker.add(0.5, 100.0)
        tracker.add(0.9, 10.0)
        stats = tracker.get()
        self.assertEqual(stats.count, 1)
        self.assertAlmostEqual(stats.median_m or -1.0, 0.9)

    def test_reset_discards_state(self) -> None:
        tracker = TemporalStatistics(window_size=5, min_valid_frames=1)
        tracker.add(0.5, 1.0)
        tracker.add_invalid(2.0)
        tracker.reset()
        stats = tracker.get()
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.invalid_streak, 0)
        self.assertFalse(stats.stale)


if __name__ == "__main__":
    unittest.main()
