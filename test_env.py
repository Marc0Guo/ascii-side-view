"""Determinism + policy discrimination tests for ascii-side-view."""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import AsciiSideViewEnv, CATALOG, MUG_PUZZLE, score_side_view  # noqa: E402


def _run_episode(env: AsciiSideViewEnv, seed: int, action_fn) -> tuple[float, dict]:
    obs = env.reset(seed=seed)
    result = env.step(action_fn(obs))
    return result.reward, result.info


def test_determinism() -> None:
    env = AsciiSideViewEnv()
    for seed in range(50):
        obs1 = env.reset(seed=seed)
        obs2 = env.reset(seed=seed)
        assert obs1 == obs2, f"reset not deterministic for seed={seed}"

        r1 = env.step("garbage")
        env2 = AsciiSideViewEnv()
        env2.reset(seed=seed)
        r2 = env2.step("garbage")
        assert r1.reward == r2.reward
        assert r1.info == r2.info
    print("determinism: OK (50 seeds)")


def test_info_strings() -> None:
    env = AsciiSideViewEnv()
    for seed in range(30):
        env.reset(seed=seed)
        r = env.step("·#·")
        for k, v in r.info.items():
            assert isinstance(v, str), f"info[{k}] is {type(v)}, not str (seed={seed})"
    print("info-all-strings: OK (30 seeds)")


def test_smart_vs_dumb() -> None:
    env = AsciiSideViewEnv()
    smart_rewards: list[float] = []
    dumb_rewards: list[float] = []

    for seed in range(20):
        obs = env.reset(seed=seed)
        assert env._puzzle is not None
        expected = env._puzzle.side
        front = obs["front_view"]

        smart_rewards.append(env.step(expected).reward)

        env.reset(seed=seed)
        dumb_rewards.append(env.step(front).reward)

    smart_mean = sum(smart_rewards) / len(smart_rewards)
    dumb_mean = sum(dumb_rewards) / len(dumb_rewards)
    print(f"smart mean step-reward: {smart_mean:.3f}")
    print(f"dumb  mean step-reward: {dumb_mean:.3f}")
    assert smart_mean > 0.85, "exact side view should score high"
    assert dumb_mean < smart_mean - 0.1, "front-view mirror must score worse"
    print("smart-vs-dumb: OK")


def test_different_prompts_per_seed() -> None:
    env = AsciiSideViewEnv()
    seen: set[str] = set()
    for seed in range(min(12, len(CATALOG))):
        obs = env.reset(seed=seed)
        seen.add(obs["front_view"])
    assert len(seen) >= 2, "seeds should pick different objects from catalog"
    print(f"different-prompts-per-seed: OK ({len(seen)} unique fronts in 12 seeds)")


def test_iou_delta_multistep() -> None:
    env = AsciiSideViewEnv()
    env.reset(seed=0)
    r1 = env.step(MUG_PUZZLE.front)
    assert float(r1.info["iou_delta"]) == float(r1.info["iou"])
    if not r1.terminated:
        r2 = env.step(MUG_PUZZLE.side)
        assert float(r2.info["iou_delta"]) >= 0
        assert float(r2.info["iou"]) > float(r1.info["iou"])
    print("iou-delta-multistep: OK")


def test_illegal_char_penalty() -> None:
    from env import evaluate_submission

    stats = evaluate_submission("hello\n.##.", MUG_PUZZLE.side)
    assert stats["illegal_count"] > 0
    assert stats["illegal_penalty"] > 0
    print("illegal-char-penalty: OK")


def test_mug_example() -> None:
    reward, stats = score_side_view(MUG_PUZZLE.side, MUG_PUZZLE.side)
    assert reward >= 0.99 and stats["exact"]
    reward_front, _ = score_side_view(MUG_PUZZLE.front, MUG_PUZZLE.side)
    assert reward_front < 0.85
    print("mug example: OK")


def test_prose_stripped_from_parse() -> None:
    from env import _parse_ascii, _grid_to_text

    messy = (
        "The front view shows a base of width 5 and a tower of width 3.\n"
        "Rotating this 90 degrees to the right...\n\n"
        ".......\n"
        ".#####.\n"
        ".#####.\n"
        "..###..\n"
        "......."
    )
    grid = _parse_ascii(messy)
    text = _grid_to_text(grid)
    assert "The" not in text
    assert ".#####." in text
    assert len(grid) == 5
    print("prose-stripped parse: OK")


def test_catalog_nonempty() -> None:
    assert len(CATALOG) >= 10
    for p in CATALOG:
        assert p.front and p.side and p.description
        assert "#" in p.front and "#" in p.side
    print(f"catalog: OK ({len(CATALOG)} puzzles)")


if __name__ == "__main__":
    test_catalog_nonempty()
    test_mug_example()
    test_prose_stripped_from_parse()
    test_different_prompts_per_seed()
    test_illegal_char_penalty()
    test_iou_delta_multistep()
    test_determinism()
    test_info_strings()
    test_smart_vs_dumb()
    print("\nALL TESTS PASSED")
