"""Determinism + policy discrimination tests for ascii-side-view."""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import AsciiSideViewEnv, CATALOG, score_side_view  # noqa: E402


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

    for seed in range(40):
        obs = env.reset(seed=seed)
        expected = env._puzzle.side  # noqa: SLF001 — test access
        front = obs["front_view"]

        smart_rewards.append(env.step(expected).reward)

        env.reset(seed=seed)
        dumb_rewards.append(env.step(front).reward)

    smart_mean = sum(smart_rewards) / len(smart_rewards)
    dumb_mean = sum(dumb_rewards) / len(dumb_rewards)
    print(f"smart mean reward: {smart_mean:.3f}")
    print(f"dumb  mean reward: {dumb_mean:.3f}")
    assert smart_mean > 0.9, "smart policy should score near-perfect"
    assert dumb_mean < smart_mean - 0.15, "dumb policy must score meaningfully worse"
    print("smart-vs-dumb: OK")


def test_mug_example() -> None:
    mug = next(p for p in CATALOG if p.name == "mug")
    reward, stats = score_side_view(mug.side, mug.side)
    assert reward >= 0.99 and stats["exact"]
    reward_front, _ = score_side_view(mug.front, mug.side)
    assert reward_front < 0.85
    print("mug example: OK")


def test_catalog_nonempty() -> None:
    assert len(CATALOG) >= 10
    for p in CATALOG:
        assert p.front and p.side and p.description
        assert "#" in p.front and "#" in p.side
    print(f"catalog: OK ({len(CATALOG)} puzzles)")


if __name__ == "__main__":
    test_catalog_nonempty()
    test_mug_example()
    test_determinism()
    test_info_strings()
    test_smart_vs_dumb()
    print("\nALL TESTS PASSED")
