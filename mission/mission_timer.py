"""
Timing of the exploration experiment.

Two kinds of measurements, both on the monotonic clock (immune to
wall-clock adjustments during the mission):

    PHASES: every timed block (acquisition, mapping, planning,
    navigation, return_home, ...) is measured with

        with timer.measure("planning") as span:
            ...
        span.duration   # seconds of this block

    and accumulated per phase name: total time, number of runs, mean,
    min, max. Blocks can be nested (e.g. "return_navigation" inside
    "return_home"), each counted in its own phase, so phase totals are
    NOT meant to add up to the mission time.

    Phases used by ExplorationMission:
        init                 lease, power on, stand, acquirer setup
        acquisition          capture of all cameras (exploration)
        logging              snapshot written to disk
        mapping              SegFormer + point clouds + GridMap fusion
                             (stop snapshot + those taken on the move)
        planning             C_total + frontier planning
        plotting             per-cycle planning plot
        navigation           walking to the frontier (includes the
                             moving captures made on the way)
        moving_acquisition   one capture on the move (moving_capture)
        cycle                one whole exploration cycle
        return_home          the whole return home, of which:
          return_acquisition / return_mapping / return_planning /
          return_navigation
        shutdown             sit, power off, lease return

    MARKS: named instants (mission_start, exploration_start,
    exploration_end, return_home_end, mission_end, ...), from which
    the intervals of the summary are computed. Each mark also stores
    the wall-clock time (ISO 8601) for the report.

The intervals reported by summary():
    init_s          mission_start      -> exploration_start
                    (connection, lease, power on, stand, setup)
    exploration_s   exploration_start  -> exploration_end
                    (the time compared with max_exploration_time_s)
    return_home_s   exploration_end    -> return_home_end
    shutdown_s      return_home_end    -> mission_end
                    (or exploration_end -> mission_end without return)
    total_s         mission_start      -> mission_end
Missing marks (e.g. a mission aborted during INIT) give None.
"""

import time
from contextlib import contextmanager
from datetime import datetime


class Span:
    """Result of one measure() block."""

    def __init__(self, phase):
        self.phase = phase
        self.start = None
        self.duration = None


class MissionTimer:

    def __init__(self):
        self.phases = {}
        self.marks = {}
        self.wall_marks = {}

    # -----------------------------------------------------------------
    # Measuring
    # -----------------------------------------------------------------

    @contextmanager
    def measure(self, phase):
        """Time a block; the time is added to `phase` even if it raises."""

        span = Span(phase)
        span.start = time.monotonic()

        try:
            yield span
        finally:
            span.duration = time.monotonic() - span.start
            self.add(phase, span.duration)

    def add(self, phase, duration):
        stats = self.phases.setdefault(phase, {
            "total_s": 0.0, "count": 0, "min_s": float("inf"), "max_s": 0.0,
        })

        stats["total_s"] += duration
        stats["count"] += 1
        stats["min_s"] = min(stats["min_s"], duration)
        stats["max_s"] = max(stats["max_s"], duration)

    def mark(self, name):
        """Record a named instant (overwrites a previous mark of that name)."""

        self.marks[name] = time.monotonic()
        self.wall_marks[name] = datetime.now().isoformat(timespec="seconds")

    def has_mark(self, name):
        return name in self.marks

    def elapsed_since(self, name):
        """Seconds since mark `name`, None if it was never set."""

        if name not in self.marks:
            return None

        return time.monotonic() - self.marks[name]

    def interval(self, start_mark, end_mark):
        """Seconds between two marks, None if either is missing."""

        if start_mark not in self.marks or end_mark not in self.marks:
            return None

        return self.marks[end_mark] - self.marks[start_mark]

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    def phase_summary(self):
        summary = {}

        for phase, stats in self.phases.items():
            summary[phase] = {
                "total_s": round(stats["total_s"], 3),
                "count": stats["count"],
                "mean_s": round(stats["total_s"] / stats["count"], 3),
                "min_s": round(stats["min_s"], 3),
                "max_s": round(stats["max_s"], 3),
            }

        return summary

    def summary(self):
        shutdown_start = (
            "return_home_end" if "return_home_end" in self.marks else "exploration_end"
        )

        intervals = {
            "init_s": self.interval("mission_start", "exploration_start"),
            "exploration_s": self.interval("exploration_start", "exploration_end"),
            "return_home_s": self.interval("exploration_end", "return_home_end"),
            "shutdown_s": self.interval(shutdown_start, "mission_end"),
            "total_s": self.interval("mission_start", "mission_end"),
        }

        return {
            "intervals": {
                key: (None if value is None else round(value, 3))
                for key, value in intervals.items()
            },
            "wall_clock": dict(self.wall_marks),
            "phases": self.phase_summary(),
        }

    def format_summary(self):
        """Human-readable table of summary()."""

        summary = self.summary()
        lines = ["Timing"]

        for key, value in summary["intervals"].items():
            label = key[:-2].replace("_", " ")
            lines.append(f"  {label:<14} {self.format_duration(value)}")

        lines.append("")
        lines.append(f"  {'phase':<16}{'total':>10}{'runs':>6}{'mean':>9}{'min':>9}{'max':>9}")

        for phase, stats in summary["phases"].items():
            lines.append(
                f"  {phase:<16}{self.format_duration(stats['total_s']):>10}{stats['count']:>6}"
                f"{stats['mean_s']:>8.2f}s{stats['min_s']:>8.2f}s{stats['max_s']:>8.2f}s"
            )

        return "\n".join(lines)

    @staticmethod
    def format_duration(seconds):
        if seconds is None:
            return "n/a"

        minutes, seconds = divmod(float(seconds), 60.0)

        if minutes >= 60:
            hours, minutes = divmod(minutes, 60.0)
            return f"{int(hours)}h{int(minutes):02d}m{seconds:04.1f}s"

        if minutes >= 1:
            return f"{int(minutes)}m{seconds:04.1f}s"

        return f"{seconds:.2f}s"