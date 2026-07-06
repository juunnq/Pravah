"""Barrier configuration for crowd safety optimization."""

from dataclasses import dataclass

import numpy as np

from sim.core.world import Obstacle


@dataclass
class BarrierConfig:
    """Configurable barrier (rotated rectangle obstacle).

    Args:
        x: Center x-coordinate.
        y: Center y-coordinate.
        length: Barrier length (m).
        angle: Rotation angle (radians).
        width: Barrier width (m).
    """

    x: float
    y: float
    length: float
    angle: float
    width: float = 0.3

    def to_obstacle(self) -> Obstacle:
        """Convert to a rotated rectangle Obstacle.

        Returns:
            Obstacle with 4 vertices.
        """
        dx = np.array([np.cos(self.angle), np.sin(self.angle)])
        dy = np.array([-np.sin(self.angle), np.cos(self.angle)])
        center = np.array([self.x, self.y])

        hl = self.length / 2.0
        hw = self.width / 2.0

        vertices = np.array([
            center - hl * dx - hw * dy,
            center + hl * dx - hw * dy,
            center + hl * dx + hw * dy,
            center - hl * dx + hw * dy,
        ])
        return Obstacle(vertices)

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Parameter bounds for optimization: [x, y, length, angle].

        Returns:
            List of (min, max) tuples for each parameter.
        """
        return [
            (2.0, 9.0),    # x
            (2.0, 8.0),    # y
            (0.5, 3.0),    # length
            (0.0, np.pi),  # angle
        ]
