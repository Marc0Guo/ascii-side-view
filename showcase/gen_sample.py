"""Generate sample replay — 7 episodes × 7 steps each (refinement trajectory)."""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from env import AsciiSideViewEnv, CATALOG, evaluate_submission  # noqa: E402

MODEL = "gemini/gemini-3.1-flash-lite"
NUM_EPISODES = 7
STEPS_PER_EP = 7


def _imperfect_submission(p, step: int) -> str:
    """Pick a submission that is not an exact match (keeps episode at 7 steps)."""
    lines = p.side.splitlines()
    candidates = [
        f"step {step}\n{p.front}",
        f"step {step}\n" + "\n".join(lines[: max(1, len(lines) // 2)]),
        p.side.replace("#", ".", 1),
        ".\n.\n.",
    ]
    for c in candidates:
        if not evaluate_submission(c, p.side)["exact"]:
            return c
    return f"step {step}\n."


def _submission_for_step(p, step: int) -> str:
    if step >= STEPS_PER_EP:
        return p.side
    return _imperfect_submission(p, step)


def main() -> None:
    episodes_meta: list[dict] = []
    replay: dict[str, list] = {}
    total_ep_reward = 0.0
    final_ious: list[float] = []

    for i in range(NUM_EPISODES):
        env = AsciiSideViewEnv()
        obs = env.reset(seed=i, max_steps=STEPS_PER_EP)
        assert env._puzzle is not None
        p = env._puzzle
        ep_id = f"ep-{i + 1:02d}"
        turns: list[dict] = []
        ep_reward = 0.0

        for step in range(1, STEPS_PER_EP + 1):
            act = _submission_for_step(p, step)
            rs = env.step(act)
            ep_reward += rs.reward
            turns.append({
                "step": step,
                "timestamp": f"2026-05-30T16:{i:02d}:{step:02d}+00:00",
                "observation": obs,
                "reasoning": f"Episode {i + 1} step {step}/{STEPS_PER_EP}: {p.description}",
                "model": MODEL,
                "action": act,
                "reward": round(rs.reward, 4),
                "terminated": rs.terminated,
                "truncated": rs.truncated,
                "info": dict(rs.info),
                "episode_end": (
                    {
                        "total_reward": round(ep_reward, 4),
                        "steps": step,
                        "status": "completed",
                        "terminal_info": dict(rs.info),
                    }
                    if rs.terminated or rs.truncated
                    else None
                ),
            })
            obs = rs.observation
            if rs.terminated or rs.truncated:
                break

        replay[ep_id] = turns
        final_iou = float(turns[-1]["info"]["final_iou"])
        final_ious.append(final_iou)
        total_ep_reward += ep_reward
        episodes_meta.append({
            "id": ep_id,
            "seed": i,
            "status": "completed",
            "total_reward": round(ep_reward, 4),
            "steps": len(turns),
        })

    export = {
        "schema_version": "1",
        "exported_at": "2026-05-30T16:30:00+00:00",
        "visibility": "gallery_public",
        "domain_id": "ascii-side-view",
        "domain_name": "Ascii Side View — 7 episodes × 7 steps",
        "binding_vow_version": "1.0.0",
        "run": {
            "id": "sample-run",
            "config": {
                "domain_id": "ascii-side-view",
                "binding_vow_version": "1.0.0",
                "agent_config": {"model": MODEL},
                "num_episodes": NUM_EPISODES,
            },
            "status": "completed",
            "scores": {
                "mean_iou": round(sum(final_ious) / NUM_EPISODES, 4),
                "exact_match_rate": round(
                    sum(1 for x in final_ious if x >= 0.999) / NUM_EPISODES, 4
                ),
                "mean_episode_reward": round(total_ep_reward / NUM_EPISODES, 4),
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
    names = [CATALOG[i % len(CATALOG)].name for i in range(NUM_EPISODES)]
    steps = [len(replay[k]) for k in replay]
    print(f"sample: {NUM_EPISODES} eps × {STEPS_PER_EP} steps, objects={names}, steps={steps}")


if __name__ == "__main__":
    main()
