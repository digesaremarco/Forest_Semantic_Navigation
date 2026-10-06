"""
Blacklist of exploration targets the robot failed to reach.

When navigation to a frontier fails (Spot stalls, the waypoint times
out, the terrain turns out impassable), the main loop calls
PlanningPipeline.blacklist_target() with that frontier's target; from
then on, plan() discards every frontier candidate whose target lies
within blacklist_radius_m of a blacklisted point, so the next cycle
naturally falls back to the next-best frontier instead of retrying
the same one forever.

Stored in world-frame meters (vision), not grid cells: the points stay
valid whatever happens to the grid, and they are exactly what the
main loop has at hand (the goal it was navigating to).
"""

import numpy as np


class TargetBlacklist:

    def __init__(self, radius_m):
        if radius_m <= 0:
            raise ValueError(f"radius_m must be > 0, got {radius_m}.")

        self.radius_m = float(radius_m)
        self.points = []

    def add(self, position_world):
        """Blacklist a world-frame (x, y) target [m]."""

        self.points.append((float(position_world[0]), float(position_world[1])))

    def contains(self, position_world):
        """Whether (x, y) lies within radius_m of a blacklisted point."""

        if not self.points:
            return False

        points = np.asarray(self.points, dtype=np.float64)
        position = np.asarray(position_world[:2], dtype=np.float64)

        return bool(np.any(np.linalg.norm(points - position, axis=1) <= self.radius_m))

    def clear(self):
        self.points = []

    def __len__(self):
        return len(self.points)
