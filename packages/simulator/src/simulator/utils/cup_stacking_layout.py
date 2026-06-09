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

    def expanded_xy(self, margin: float) -> UniformXYSpawn:
        """Return the spawn range expanded by ``margin`` on every xy edge."""
        if margin < 0.0:
            raise ValueError("XY expansion margin must be non-negative.")
        return UniformXYSpawn(
            center=self.center,
            x_offset=(self.x_offset[0] - margin, self.x_offset[1] + margin),
            y_offset=(self.y_offset[0] - margin, self.y_offset[1] + margin),
            quat_wxyz=self.quat_wxyz,
        )


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

CUP_STACKING_EVAL_WIDE_MARGIN = 0.02
CUP_STACKING_EVAL_WIDE_SPAWNS: dict[str, UniformXYSpawn] = {
    name: spawn.expanded_xy(CUP_STACKING_EVAL_WIDE_MARGIN)
    for name, spawn in CUP_STACKING_EVAL_SPAWNS.items()
}


def sample_cup_stacking_layout(
    rng: random.Random,
    spawns: dict[str, UniformXYSpawn],
) -> dict[str, WorldPose]:
    """Sample independent cup positions from the selected spawn distribution."""
    return {name: spawn.sample(rng) for name, spawn in spawns.items()}


def sample_cup_stacking_eval_layout(rng: random.Random) -> dict[str, WorldPose]:
    """Sample the exact independent cup-position distribution used by evaluation."""
    return sample_cup_stacking_layout(rng, CUP_STACKING_EVAL_SPAWNS)


def sample_cup_stacking_eval_wide_layout(rng: random.Random) -> dict[str, WorldPose]:
    """Sample the evaluation distribution with a small reach-safe xy margin."""
    return sample_cup_stacking_layout(rng, CUP_STACKING_EVAL_WIDE_SPAWNS)
