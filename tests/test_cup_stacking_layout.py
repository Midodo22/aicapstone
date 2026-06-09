import random

import pytest

from simulator.utils.cup_stacking_layout import (
    CUP_STACKING_EVAL_SPAWNS,
    sample_cup_stacking_eval_layout,
)


def test_eval_spawn_ranges_match_fixed_evaluation_config():
    assert CUP_STACKING_EVAL_SPAWNS["blue_cup"].x_range == pytest.approx((0.31, 0.41))
    assert CUP_STACKING_EVAL_SPAWNS["blue_cup"].y_range == pytest.approx((-0.45, -0.35))
    assert CUP_STACKING_EVAL_SPAWNS["pink_cup"].x_range == pytest.approx((0.41, 0.51))
    assert CUP_STACKING_EVAL_SPAWNS["pink_cup"].y_range == pytest.approx((-0.45, -0.35))


def test_eval_layout_samples_independent_cup_positions_with_fixed_pose():
    rng = random.Random(42)

    for _ in range(100):
        layout = sample_cup_stacking_eval_layout(rng)
        assert set(layout) == {"blue_cup", "pink_cup"}

        for name, (pos, quat) in layout.items():
            spawn = CUP_STACKING_EVAL_SPAWNS[name]
            assert spawn.x_range[0] <= pos[0] <= spawn.x_range[1]
            assert spawn.y_range[0] <= pos[1] <= spawn.y_range[1]
            assert pos[2] == pytest.approx(0.12)
            assert quat == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_eval_layout_is_seed_reproducible():
    assert sample_cup_stacking_eval_layout(random.Random(7)) == sample_cup_stacking_eval_layout(random.Random(7))
