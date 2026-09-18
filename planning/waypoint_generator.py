"""
Waypoint generation: converts a smoothed grid-cell path into a
sequence of waypoints in world-frame meters, resampled to roughly
waypoint_spacing_m apart, ready to hand to Spot's own locomotion
controller.

Takes plain resolution/origin/cell_n rather than a live GridMap
reference -- consistent with the rest of planning/, which works on
plain arrays and scalars rather than live pipeline objects, so each
module stays independently testable. The cell-to-world formula here
matches GridMap.cell_to_world() exactly.

Resampling is by ARC LENGTH along the smoothed path's polyline (sum
of straight-line segment lengths), not by index -- so waypoint
spacing stays close to waypoint_spacing_m regardless of how uneven
the spacing between the smoothed path's own points is.

Each waypoint also carries a heading (yaw, radians), pointing toward
the NEXT waypoint -- the last waypoint repeats the heading of the
one before it, since there is no further point to face. This is
included because most mobile-robot controllers expect a target
orientation alongside a target position; if Spot's controller turns
out not to need it, it can simply be ignored.
"""

import numpy as np


class WaypointGenerator:

    def __init__(self, planning_config):
        self.planning_config = planning_config

    def generate(self, path, resolution, origin, cell_n):
        """
        Parameters
        ----------
        path : list of (row, col) int tuples, or None
            As returned by PathSmoother.smooth() (or PathPlanner.plan()
            directly, if smoothing is skipped).

        resolution : float
            Grid cell size, in meters.

        origin : numpy.ndarray or tuple
            (x, y), the world position the grid's center cell is
            anchored to -- GridMap.get_origin()[:2].

        cell_n : int
            Number of cells per side of the (square) grid.

        Returns
        -------
        list of dict, or None if `path` was None. Each dict has:
            "position" : (x, y) tuple, meters.
            "heading" : float, radians, pointing toward the next
                waypoint (or the previous heading, for the last one).
        Empty list if `path` has fewer than 2 points (nothing to
        navigate to).
        """

        if path is None:
            return None

        if len(path) < 2:
            return []

        world_points = [
            self.cell_to_world(row, col, resolution, origin, cell_n)
            for row, col in path
        ]

        resampled_points = self.resample_by_arc_length(
            world_points, self.planning_config.waypoint_spacing_m
        )

        return self.attach_headings(resampled_points)

    @staticmethod
    def cell_to_world(row, col, resolution, origin, cell_n):
        half = cell_n // 2

        x = (col - half) * resolution + origin[0]
        y = (row - half) * resolution + origin[1]

        return (float(x), float(y))

    @staticmethod
    def resample_by_arc_length(points, spacing):
        """
        Resample a polyline (list of (x, y) tuples) to points spaced
        `spacing` meters apart along its length, always including
        the first and last original point exactly.
        """

        points = np.array(points, dtype=np.float64)

        if len(points) < 2:
            return [tuple(p) for p in points]

        segment_vectors = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)

        cumulative_lengths = np.concatenate(
            [[0.0], np.cumsum(segment_lengths)]
        )

        total_length = cumulative_lengths[-1]

        if total_length == 0:
            return [tuple(points[0])]

        num_waypoints = max(2, int(np.floor(total_length / spacing)) + 1)
        target_distances = np.linspace(0.0, total_length, num_waypoints)

        resampled = []

        for target in target_distances:
            idx = np.searchsorted(cumulative_lengths, target)
            idx = min(idx, len(cumulative_lengths) - 1)

            if idx == 0:
                resampled.append(tuple(points[0]))
                continue

            segment_start_length = cumulative_lengths[idx - 1]
            segment_length = segment_lengths[idx - 1]

            if segment_length == 0:
                resampled.append(tuple(points[idx]))
                continue

            t = (target - segment_start_length) / segment_length
            t = np.clip(t, 0.0, 1.0)

            point = points[idx - 1] + t * (points[idx] - points[idx - 1])

            resampled.append((float(point[0]), float(point[1])))

        return resampled

    @staticmethod
    def attach_headings(points):
        """
        Pair each waypoint with a heading (yaw, radians) pointing
        toward the next one; the last waypoint repeats the previous
        heading.
        """

        if len(points) == 0:
            return []

        waypoints = []

        for i in range(len(points)):
            if i < len(points) - 1:
                dx = points[i + 1][0] - points[i][0]
                dy = points[i + 1][1] - points[i][1]
                heading = float(np.arctan2(dy, dx))
            else:
                heading = waypoints[-1]["heading"] if waypoints else 0.0

            waypoints.append({
                "position": points[i],
                "heading": heading
            })

        return waypoints