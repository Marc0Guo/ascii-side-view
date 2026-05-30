"""Generate sample replay for showcase preview. Replace with:
    mesocosm run export RUN_ID -o showcase/data/replay.json
"""

from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from env import AsciiSideViewEnv, CATALOG  # noqa: E402

MODEL = "gemini/gemini-3.1-flash-lite"


def mock_reasoning(obs: dict, action_preview: str, correct: bool) -> str:
    desc = obs["description"]
    if correct:
        return (
            f"The object is {desc}. From the front I see width and height; rotating 90° "
            f"to the right collapses depth onto the horizontal axis. Features that stick "
            f"out left/right in the front view (like a handle) disappear in the side silhouette. "
            f"I project the voxel occupancy along the depth axis."
        )
    return (
        f"This is {desc}. I'll try mirroring the front view, though side view should "
        f"show depth not width — I may be confusing axes."
    )


def policy(seed: int, puzzle_name: str, side: str, front: str, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.35:
        return side
    if roll < 0.55:
        return front
    if roll < 0.75:
        return side.replace("#", ".")[: len(side) // 2] + side[len(side) // 2 :]
    return "\n".join("." * len(front.splitlines()) for _ in front.splitlines())


def main() -> None:
    episodes_meta: list[dict] = []
    replay: dict[str, list] = {}
    total_reward = 0.0
    exact = 0
    n = 12

    for i in range(n):
        env = AsciiSideViewEnv()
        obs = env.reset(seed=100 + i)
        rng = random.Random(500 + i)
        ep_id = f"ep-{i:03d}"
        assert env._puzzle is not None
        p = env._puzzle
        act = policy(100 + i, p.name, p.side, p.front, rng)
        rs = env.step(act)
        total_reward += rs.reward
        if rs.info["exact_numeric"] == "1":
            exact += 1

        turns = [{
            "step": 1,
            "timestamp": f"2026-05-30T12:{i:02d}:01+00:00",
            "observation": obs,
            "reasoning": mock_reasoning(obs, act[:40], rs.info["exact_numeric"] == "1"),
            "model": MODEL,
            "action": act,
            "reward": round(rs.reward, 4),
            "terminated": rs.terminated,
            "truncated": rs.truncated,
            "info": dict(rs.info),
            "episode_end": {
                "total_reward": round(rs.reward, 4),
                "steps": 1,
                "status": "completed",
                "terminal_info": dict(rs.info),
            },
        }]
        replay[ep_id] = turns
        episodes_meta.append({
            "id": ep_id,
            "seed": 100 + i,
            "status": "completed",
            "total_reward": round(rs.reward, 4),
            "steps": 1,
        })

    export = {
        "schema_version": "1",
        "exported_at": "2026-05-30T12:00:00+00:00",
        "visibility": "gallery_public",
        "domain_id": "ascii-side-view",
        "domain_name": "Ascii Side View — 3D rotation from front to side",
        "binding_vow_version": "1.0.0",
        "run": {
            "id": "sample-run",
            "config": {
                "domain_id": "ascii-side-view",
                "binding_vow_version": "1.0.0",
                "agent_config": {"model": MODEL},
                "num_episodes": n,
            },
            "status": "completed",
            "scores": {
                "mean_iou": round(total_reward / n, 4),
                "exact_match_rate": round(exact / n, 4),
                "mean_episode_reward": round(total_reward / n, 4),
            },
        },
        "episodes": episodes_meta,
        "traces": {},
        "replay": replay,
    }
    data_dir = os.path.join(HERE, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "replay.json"), "w") as f:
        json.dump(export, f, indent=2)
    with open(os.path.join(data_dir, "replay.js"), "w") as f:
        f.write("window.REPLAY = ")
        json.dump(export, f)
        f.write(";\n")
    print(f"sample: {n} episodes, exact={exact}/{n}, mean_reward={total_reward/n:.3f}")


if __name__ == "__main__":
    main()
