"""Generate 7-run demo replay — same mug prompt, gemini-style varied outputs."""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from env import AsciiSideViewEnv, MUG_PUZZLE  # noqa: E402

MODEL = "gemini/gemini-3.1-flash-lite"
N = 7

# Simulated model outputs (prose + wrong size bugs from real run patterns)
SUBMISSIONS = [
    (
        "The mug is cylindrical; handle hidden from side.\n\n"
        "..........\n..######..\n.########.\n.########.\n.########.\n"
        ".########.\n.########.\n.######...\n.........."
    ),
    MUG_PUZZLE.front,  # classic bug: mirrors front view
    (
        "..........\n·######···\n·######···\n·######···\n·######···\n"
        "·######···\n·######···\n·######···\n··········\n··········"
    ),
    (
        "Side view:\n"
        "..........\n.######...\n.######...\n.######...\n.######...\n"
        ".######...\n.######...\n.######...\n..........\n.........."
    ),
    (
        "..........\n.######...\n.######...\n.######...\n.######...\n"
        ".######...\n.######...\n.######...\n..........\n.........."
    ),
    (
        "Rotating 90 right collapses handle.\n"
        "####......\n####......\n####......\n####......\n"
        "####......\n####......\n####......\n..........\n..........\n.........."
    ),
    MUG_PUZZLE.side,
]


def main() -> None:
    episodes_meta: list[dict] = []
    replay: dict[str, list] = {}
    total_reward = 0.0
    ious: list[float] = []

    for i in range(N):
        env = AsciiSideViewEnv()
        obs = env.reset(seed=i)
        ep_id = f"run-{i + 1:02d}"
        act = SUBMISSIONS[i % len(SUBMISSIONS)]
        rs = env.step(act)
        total_reward += rs.reward
        final_iou = float(rs.info["final_iou"])
        ious.append(final_iou)

        turns = [{
            "step": 1,
            "timestamp": f"2026-05-30T15:{i:02d}:01+00:00",
            "observation": obs,
            "reasoning": f"Run {i + 1}/7 on the fixed mug prompt (seed={i}).",
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
            "seed": i,
            "status": "completed",
            "total_reward": round(rs.reward, 4),
            "steps": 1,
        })

    export = {
        "schema_version": "1",
        "exported_at": "2026-05-30T15:00:00+00:00",
        "visibility": "gallery_public",
        "domain_id": "ascii-side-view",
        "domain_name": "Ascii Side View — same mug prompt × 7 runs",
        "binding_vow_version": "1.0.0",
        "run": {
            "id": "demo-7-runs",
            "config": {
                "domain_id": "ascii-side-view",
                "binding_vow_version": "1.0.0",
                "agent_config": {"model": MODEL},
                "num_episodes": N,
            },
            "status": "completed",
            "scores": {
                "mean_iou": round(sum(ious) / N, 4),
                "exact_match_rate": round(sum(1 for x in ious if x >= 0.999) / N, 4),
                "mean_episode_reward": round(total_reward / N, 4),
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
    print(f"demo: {N} runs same mug, mean_iou={sum(ious)/N:.3f}")


if __name__ == "__main__":
    main()
