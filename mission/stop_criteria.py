"""
Exploration stop criteria and map coverage.

COVERAGE: observed cells (GridMap count > 0) over the cells of the
EXPLORABLE region -- the grid minus the border band of
border_margin_cells that FrontierDetector ignores (planning_config.yaml).
Cells in that band can never become frontiers, so the robot is never
sent to observe them: counting them would make the threshold
unreachable by construction. Observed cells inside the band still do
not count, for the same reason.

CRITERIA, checked once per cycle after the map update -- the first one
met ends the exploration (the robot then returns home):

    coverage            coverage >= coverage_threshold
    time                exploration time >= max_exploration_time_s
    battery             battery < min_battery_percent
    nav_failures        consecutive navigation failures
                        >= max_consecutive_nav_failures
    acquisition_failures consecutive acquisition failures
                        >= max_consecutive_acquisition_failures

"No frontier left" is not checked here: it comes out of the planner
(PlanningPipeline.STATUS_NO_FRONTIERS).
"""

import numpy as np


class ExplorationCoverage:

    def __init__(self, border_margin_cells):
        self.border_margin_cells = int(border_margin_cells)
        self.region_mask = None

    def compute(self, count_layer):
        """
        Parameters
        ----------
        count_layer : numpy.ndarray
            (H, W) GridMap.get_count_layer().

        Returns
        -------
        dict: "coverage" (fraction), "observed_cells", "region_cells".
        """

        if self.region_mask is None or self.region_mask.shape != count_layer.shape:
            self.region_mask = self.build_region_mask(count_layer.shape)

        region_cells = int(self.region_mask.sum())
        observed_cells = int(np.sum((count_layer > 0) & self.region_mask))

        return {
            "coverage": observed_cells / region_cells if region_cells else 0.0,
            "observed_cells": observed_cells,
            "region_cells": region_cells,
        }

    def build_region_mask(self, shape):
        mask = np.zeros(shape, dtype=bool)
        margin = self.border_margin_cells

        if 2 * margin >= min(shape):
            raise ValueError(
                f"border_margin_cells ({margin}) leaves no explorable region "
                f"in a {shape} grid."
            )

        mask[margin:shape[0] - margin, margin:shape[1] - margin] = True

        return mask


class StopCriteria:

    COVERAGE = "coverage_reached"
    TIME = "max_exploration_time"
    BATTERY = "low_battery"
    NAV_FAILURES = "too_many_navigation_failures"
    ACQUISITION_FAILURES = "too_many_acquisition_failures"

    def __init__(self, mission_config):
        self.config = mission_config

    def check(self, coverage, exploration_time_s, battery_percent,
              consecutive_nav_failures, consecutive_acquisition_failures):
        """
        Returns
        -------
        (reason, detail) of the first criterion met, or (None, None).
        battery_percent None (not reported) is ignored.
        """

        config = self.config

        if coverage is not None and coverage >= config.coverage_threshold:
            return self.COVERAGE, (
                f"coverage {coverage * 100:.1f} % >= {config.coverage_threshold * 100:.0f} %"
            )

        if exploration_time_s >= config.max_exploration_time_s:
            return self.TIME, (
                f"exploration time {exploration_time_s:.0f} s >= "
                f"{config.max_exploration_time_s:.0f} s"
            )

        if battery_percent is not None and battery_percent < config.min_battery_percent:
            return self.BATTERY, (
                f"battery {battery_percent:.0f} % < {config.min_battery_percent:.0f} %"
            )

        if consecutive_nav_failures >= config.max_consecutive_nav_failures:
            return self.NAV_FAILURES, (
                f"{consecutive_nav_failures} consecutive navigation failures"
            )

        if consecutive_acquisition_failures >= config.max_consecutive_acquisition_failures:
            return self.ACQUISITION_FAILURES, (
                f"{consecutive_acquisition_failures} consecutive acquisition failures"
            )

        return None, None
