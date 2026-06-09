"""Cup-stacking start-pose distributions used by data generation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from simulator.utils.object_poses_loader import WorldPose


@dataclass(frozen=True)
class UniformXYSpawn:
    """An object spawn pose with independent uniform x/y offsets."""

    center: tuple[float, float, float]
    x_offset: tuple[float, float]
    y_offset: tuple[float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def x_range(self) -> tuple[float, float]:
        return (self.center[0] + self.x_offset[0], self.center[0] + self.x_offset[1])

    @property
    def y_range(self) -> tuple[float, float]:
        return (self.center[1] + self.y_offset[0], self.center[1] + self.y_offset[1])

    def sample(self, rng: random.Random) -> WorldPose:
        x = self.center[0] + rng.uniform(*self.x_offset)
        y = self.center[1] + rng.uniform(*self.y_offset)
        return ((x, y, self.center[2]), self.quat_wxyz)


# Mirrors eval/cup_stacking_eval.py without importing or modifying the private
# evaluation config. The eval reset event independently adds +/- 5 cm to each
# cup's configured initial x/y position.
CUP_STACKING_EVAL_SPAWNS: dict[str, UniformXYSpawn] = {
    "blue_cup": UniformXYSpawn(
        center=(0.36, -0.40, 0.12),
        x_offset=(-0.05, 0.05),
        y_offset=(-0.05, 0.05),
    ),
    "pink_cup": UniformXYSpawn(
        center=(0.46, -0.40, 0.12),
        x_offset=(-0.05, 0.05),
        y_offset=(-0.05, 0.05),
    ),
}


def sample_cup_stacking_eval_layout(rng: random.Random) -> dict[str, WorldPose]:
    """Sample the same independent cup-position distribution used by evaluation."""
    return {name: spawn.sample(rng) for name, spawn in CUP_STACKING_EVAL_SPAWNS.items()}
