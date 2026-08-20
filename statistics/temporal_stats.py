"""Finite-window distance statistics with explicit stale-data handling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalStats:
    """A snapshot of the current valid-distance window.

    Statistical values are ``None`` when the window is empty.  ``ready`` only
    becomes true after ``min_valid_frames`` samples have accumulated and the
    window is not stale.  This lets UI and recording code distinguish a stable
    zero-valued statistic from unavailable data without emitting NaN.
    """

    count: int
    ready: bool
    median_m: float | None
    mean_m: float | None
    std_m: float | None
    mad_m: float | None
    stale: bool = False
    invalid_streak: int = 0
    age_ms: float | None = None


class TemporalStatistics:
    """Maintain recent finite, non-negative distance measurements.

    Invalid input is never inserted.  A configurable invalid-frame streak or
    elapsed time marks old results stale and clears them, preventing the UI
    from displaying an arbitrarily old median after a QR disappears.

    Timestamps use milliseconds to match RealSense frame timestamps.  If a
    caller omits a timestamp, a monotonic process-local timestamp is used.
    """

    def __init__(
        self,
        window_size: int = 20,
        min_valid_frames: int = 10,
        *,
        max_invalid_streak: int = 10,
        stale_after_ms: float = 1_000.0,
    ) -> None:
        if isinstance(window_size, bool) or int(window_size) != window_size:
            raise ValueError("window_size must be a positive integer")
        if isinstance(min_valid_frames, bool) or int(min_valid_frames) != min_valid_frames:
            raise ValueError("min_valid_frames must be a positive integer")
        if isinstance(max_invalid_streak, bool) or int(max_invalid_streak) != max_invalid_streak:
            raise ValueError("max_invalid_streak must be a positive integer")
        window_size = int(window_size)
        min_valid_frames = int(min_valid_frames)
        max_invalid_streak = int(max_invalid_streak)
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not 1 <= min_valid_frames <= window_size:
            raise ValueError("min_valid_frames must be between 1 and window_size")
        if max_invalid_streak <= 0:
            raise ValueError("max_invalid_streak must be a positive integer")
        if not math.isfinite(stale_after_ms) or stale_after_ms <= 0.0:
            raise ValueError("stale_after_ms must be positive and finite")

        self.window_size = window_size
        self.min_valid_frames = min_valid_frames
        self.max_invalid_streak = max_invalid_streak
        self.stale_after_ms = float(stale_after_ms)
        self._samples: deque[tuple[float, float]] = deque(maxlen=window_size)
        self._invalid_streak = 0
        self._stale = False
        self._last_event_timestamp_ms: float | None = None

    @staticmethod
    def _now_ms() -> float:
        return time.monotonic() * 1_000.0

    def _timestamp(self, timestamp_ms: float | None) -> float:
        resolved = self._now_ms() if timestamp_ms is None else float(timestamp_ms)
        if not math.isfinite(resolved):
            raise ValueError("timestamp_ms must be finite")
        return resolved

    def _clear_samples(self, *, stale: bool) -> None:
        self._samples.clear()
        self._stale = stale

    def _begin_event(self, timestamp_ms: float) -> None:
        previous = self._last_event_timestamp_ms
        if previous is not None and timestamp_ms < previous:
            # A backwards device timestamp signals replay seek/restart or a new
            # source epoch.  Never combine samples from the two epochs.
            self._samples.clear()
            self._invalid_streak = 0
            self._stale = False
        elif self._samples and timestamp_ms - self._samples[-1][0] > self.stale_after_ms:
            self._clear_samples(stale=True)
        self._last_event_timestamp_ms = timestamp_ms

    def add(
        self,
        distance_m: float | None,
        timestamp_ms: float | None = None,
        *,
        valid: bool = True,
    ) -> bool:
        """Add one accepted distance, returning whether it entered the window.

        Non-finite, negative, ``None``, or explicitly invalid values are
        treated exactly like an invalid frame and do not raise.
        """

        timestamp = self._timestamp(timestamp_ms)
        self._begin_event(timestamp)
        try:
            value = float(distance_m) if distance_m is not None else float("nan")
        except (TypeError, ValueError):
            value = float("nan")
        if not valid or not math.isfinite(value) or value < 0.0:
            self._record_invalid(timestamp, event_started=True)
            return False

        # _begin_event may have marked a previous window stale.  A new valid
        # sample begins a fresh window rather than inheriting that state.
        self._stale = False
        self._invalid_streak = 0
        self._samples.append((timestamp, value))
        return True

    def _record_invalid(self, timestamp_ms: float, *, event_started: bool) -> None:
        if not event_started:
            self._begin_event(timestamp_ms)
        self._invalid_streak += 1
        if self._invalid_streak >= self.max_invalid_streak:
            self._clear_samples(stale=True)

    def add_invalid(self, timestamp_ms: float | None = None) -> None:
        """Record one invalid frame without adding a numeric sample."""

        timestamp = self._timestamp(timestamp_ms)
        self._record_invalid(timestamp, event_started=False)

    def update(
        self,
        distance_m: float | None,
        *,
        valid: bool,
        timestamp_ms: float | None = None,
    ) -> bool:
        """Explicit-validity alias convenient for a per-frame main loop."""

        return self.add(distance_m, timestamp_ms=timestamp_ms, valid=valid)

    def reset(self) -> None:
        """Start a fresh source epoch and discard every historical value."""

        self._samples.clear()
        self._invalid_streak = 0
        self._stale = False
        self._last_event_timestamp_ms = None

    def get(self, timestamp_ms: float | None = None) -> TemporalStats:
        """Return a snapshot, expiring an old window when time has advanced."""

        if timestamp_ms is not None:
            timestamp = self._timestamp(timestamp_ms)
            self._begin_event(timestamp)

        count = len(self._samples)
        age_ms: float | None = None
        if count and self._last_event_timestamp_ms is not None:
            age_ms = max(0.0, self._last_event_timestamp_ms - self._samples[-1][0])
        if count == 0:
            return TemporalStats(
                count=0,
                ready=False,
                median_m=None,
                mean_m=None,
                std_m=None,
                mad_m=None,
                stale=self._stale,
                invalid_streak=self._invalid_streak,
                age_ms=age_ms,
            )

        values = np.fromiter((value for _, value in self._samples), dtype=np.float64)
        median = float(np.median(values))
        return TemporalStats(
            count=count,
            ready=(count >= self.min_valid_frames and not self._stale),
            median_m=median,
            mean_m=float(np.mean(values)),
            std_m=float(np.std(values, ddof=0)),
            mad_m=float(np.median(np.abs(values - median))),
            stale=self._stale,
            invalid_streak=self._invalid_streak,
            age_ms=age_ms,
        )

    @property
    def snapshot(self) -> TemporalStats:
        """Property form of :meth:`get` for consumers that do not pass time."""

        return self.get()

    def __len__(self) -> int:
        return len(self._samples)


# Compatibility aliases keep the public concept discoverable while the main
# program is free to use the shorter or more explicit class name.
TemporalStatsWindow = TemporalStatistics
TemporalStatsBuffer = TemporalStatistics


__all__ = [
    "TemporalStatistics",
    "TemporalStats",
    "TemporalStatsBuffer",
    "TemporalStatsWindow",
]
