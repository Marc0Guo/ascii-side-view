"""
Ascii Side View — given a front-view ASCII art + description, predict the side view.

The agent sees a 2D front projection of a voxel object and a natural-language
description. It must output the side view (90° rotation: looking from the right).
Objects are defined as 3D voxel sets; views are max-projections onto the X–Y
(front) and Z–Y (side) planes.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult


@dataclass(frozen=True)
class Puzzle:
    name: str
    description: str
    front: str
    side: str


def _voxels_to_front(voxels: set[tuple[int, int, int]]) -> list[list[str]]:
    if not voxels:
        return [["."]]
    xs = [v[0] for v in voxels]
    ys = [v[1] for v in voxels]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w, h = max_x - min_x + 1, max_y - min_y + 1
    grid = [["."] * w for _ in range(h)]
    for x, y, _z in voxels:
        grid[y - min_y][x - min_x] = "#"
    return grid


def _voxels_to_side(voxels: set[tuple[int, int, int]]) -> list[list[str]]:
    if not voxels:
        return [["."]]
    zs = [v[2] for v in voxels]
    ys = [v[1] for v in voxels]
    min_z, max_z = min(zs), max(zs)
    min_y, max_y = min(ys), max(ys)
    w, h = max_z - min_z + 1, max_y - min_y + 1
    grid = [["."] * w for _ in range(h)]
    for _x, y, z in voxels:
        grid[y - min_y][z - min_z] = "#"
    return grid


def _grid_to_ascii(grid: list[list[str]], pad: int = 1) -> str:
    h, w = len(grid), len(grid[0]) if grid else 0
    rows = ["." * (w + 2 * pad) for _ in range(pad)]
    for row in grid:
        rows.append("." * pad + "".join(row) + "." * pad)
    rows.extend("." * (w + 2 * pad) for _ in range(pad))
    return "\n".join(rows)


def _fill_box(voxels: set[tuple[int, int, int]], x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> None:
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                voxels.add((x, y, z))


def _build_catalog() -> list[Puzzle]:
    puzzles: list[Puzzle] = []

    # Curated mug — matches the prompt example.
    puzzles.append(
        Puzzle(
            name="mug",
            description="a mug with a handle on the right",
            front=(
                "··········\n"
                "··#####···\n"
                "·########·\n"
                "·######·#·\n"
                "·######·#·\n"
                "·######·#·\n"
                "·######·#·\n"
                "·########·\n"
                "·######···\n"
                "··········"
            ),
            side=(
                "··········\n"
                "·######···\n"
                "·######···\n"
                "·######···\n"
                "·######···\n"
                "·######···\n"
                "·######···\n"
                "·######···\n"
                "··········\n"
                "··········"
            ),
        )
    )

    def add_voxel(name: str, description: str, build) -> None:
        voxels: set[tuple[int, int, int]] = set()
        build(voxels)
        puzzles.append(
            Puzzle(
                name=name,
                description=description,
                front=_grid_to_ascii(_voxels_to_front(voxels)),
                side=_grid_to_ascii(_voxels_to_side(voxels)),
            )
        )

    add_voxel(
        "solid cube",
        "a solid cube block",
        lambda v: _fill_box(v, 0, 0, 0, 3, 3, 3),
    )

    add_voxel(
        "wide slab",
        "a flat wide slab, taller than it is deep",
        lambda v: _fill_box(v, 0, 0, 0, 5, 2, 1),
    )

    add_voxel(
        "tall tower",
        "a tall narrow tower on a wider base",
        lambda v: (
            _fill_box(v, 1, 3, 1, 3, 5, 3),
            _fill_box(v, 0, 0, 0, 4, 2, 4),
        ),
    )

    add_voxel(
        "L-shape",
        "an L-shaped block (vertical bar on the left, foot extending forward)",
        lambda v: (
            _fill_box(v, 0, 0, 0, 1, 4, 2),
            _fill_box(v, 0, 4, 0, 3, 4, 2),
        ),
    )

    add_voxel(
        "step pyramid",
        "a three-step pyramid staircase",
        lambda v: (
            _fill_box(v, 1, 2, 1, 3, 3, 3),
            _fill_box(v, 0, 1, 0, 4, 1, 4),
            _fill_box(v, 0, 0, 0, 4, 0, 4),
        ),
    )

    add_voxel(
        "car",
        "a simple car: long body with a cabin bump on top toward the front",
        lambda v: (
            _fill_box(v, 0, 1, 0, 6, 2, 2),
            _fill_box(v, 2, 0, 0, 4, 0, 2),
        ),
    )

    add_voxel(
        "book",
        "a thin upright book standing on its edge",
        lambda v: _fill_box(v, 0, 0, 0, 0, 5, 3),
    )

    add_voxel(
        "stool",
        "a stool with a flat seat and four legs",
        lambda v: (
            _fill_box(v, 0, 0, 0, 3, 0, 3),
            _fill_box(v, 0, 2, 0, 0, 2, 0),
            _fill_box(v, 3, 2, 0, 3, 2, 0),
            _fill_box(v, 0, 2, 3, 0, 2, 3),
            _fill_box(v, 3, 2, 3, 3, 2, 3),
        ),
    )

    add_voxel(
        "T-shape",
        "a T-shaped block lying flat (wide top bar, stem pointing forward)",
        lambda v: (
            _fill_box(v, 0, 0, 0, 4, 0, 1),
            _fill_box(v, 1, 1, 0, 3, 3, 1),
        ),
    )

    add_voxel(
        "arch",
        "a block with a rectangular tunnel through the middle (hole along depth)",
        lambda v: (
            _fill_box(v, 0, 0, 0, 4, 4, 0),
            _fill_box(v, 0, 0, 3, 4, 4, 3),
            _fill_box(v, 0, 0, 0, 0, 4, 3),
            _fill_box(v, 4, 0, 0, 4, 4, 3),
            _fill_box(v, 0, 0, 0, 4, 0, 3),
            _fill_box(v, 0, 4, 0, 4, 4, 3),
        ),
    )

    add_voxel(
        "cross",
        "a plus-sign shape extruded in depth",
        lambda v: (
            _fill_box(v, 1, 0, 0, 3, 4, 2),
            _fill_box(v, 0, 2, 0, 4, 2, 2),
        ),
    )

    return puzzles


CATALOG = _build_catalog()
MUG_PUZZLE = CATALOG[0]  # fixed demo prompt — same front view every episode

ILLEGAL_CHAR_PENALTY = 0.02  # per letter/digit in raw output
ILLEGAL_CHAR_PENALTY_CAP = 0.25
SIZE_MISMATCH_PENALTY = 0.12


def _is_ascii_row(line: str) -> bool:
    """True if the line looks like an ASCII-art row (only # . whitespace)."""
    s = line.replace("·", ".").rstrip()
    if not s.strip():
        return False
    if re.search(r"[a-zA-Z0-9]", s):
        return False
    if not re.search(r"[#.]", s):
        return False
    return bool(re.fullmatch(r"[#.\s]+", s))


def _grid_to_text(grid: list[list[str]]) -> str:
    if not grid:
        return "."
    return "\n".join("".join(row) for row in grid)


def _parse_ascii(text: str) -> list[list[str]]:
    """Extract a grid of #/. from model output; ignore prose/explanation lines."""
    if not text:
        return [["."]]
    fenced = re.findall(r"```(?:ascii|text)?\s*\n(.*?)```", text, re.S | re.I)
    body = fenced[-1] if fenced else text

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in body.splitlines():
        if _is_ascii_row(line):
            current.append(line.replace("·", ".").rstrip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    rows = max(blocks, key=len) if blocks else []
    if not rows:
        return [["."]]
    w = max(len(r) for r in rows)
    return [list(r.ljust(w, ".")) for r in rows]


def _trim_grid(grid: list[list[str]]) -> list[list[str]]:
    if not grid:
        return [["."]]
    h, w = len(grid), len(grid[0])
    min_r, max_r = h, -1
    min_c, max_c = w, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] == "#":
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return [["."]]
    return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]


def _pad_to(grid: list[list[str]], h: int, w: int) -> list[list[str]]:
    gh = len(grid)
    gw = len(grid[0]) if grid else 0
    out = [["."] * w for _ in range(h)]
    off_r = max(0, (h - gh) // 2)
    off_c = max(0, (w - gw) // 2)
    for r in range(gh):
        for c in range(gw):
            if off_r + r < h and off_c + c < w:
                out[off_r + r][off_c + c] = grid[r][c]
    return out


def _expected_shape(expected: str) -> tuple[int, int]:
    lines = [ln for ln in expected.splitlines() if ln.strip()]
    if not lines:
        return 1, 1
    return len(lines), max(len(ln) for ln in lines)


def _count_illegal_chars(text: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9]", text))


def evaluate_submission(
    submitted: str,
    expected: str,
    prev_iou: float = 0.0,
) -> dict[str, Any]:
    """Score one submission: IoU vs target, step reward = IoU gain − penalties."""
    illegal_count = _count_illegal_chars(submitted)
    illegal_penalty = min(ILLEGAL_CHAR_PENALTY_CAP, ILLEGAL_CHAR_PENALTY * illegal_count)

    sub_block = _parse_ascii(submitted)
    exp_h, exp_w = _expected_shape(expected)
    sub_h = len(sub_block)
    sub_w = max(len(r) for r in sub_block) if sub_block else 0
    size_mismatch = (sub_h != exp_h) or (sub_w != exp_w)
    size_penalty = SIZE_MISMATCH_PENALTY if size_mismatch else 0.0

    sub = _trim_grid(sub_block)
    exp = _trim_grid(_parse_ascii(expected))
    h = max(len(sub), len(exp))
    w = max(len(sub[0]) if sub else 0, len(exp[0]) if exp else 0)
    sub_p = _pad_to(sub, h, w)
    exp_p = _pad_to(exp, h, w)

    matches = false_pos = false_neg = 0
    for r in range(h):
        for c in range(w):
            s, e = sub_p[r][c] == "#", exp_p[r][c] == "#"
            if s or e:
                if s and e:
                    matches += 1
                elif s:
                    false_pos += 1
                else:
                    false_neg += 1

    total = matches + false_pos + false_neg
    if total == 0:
        iou = 1.0 if submitted.strip() == expected.strip() else 0.0
    else:
        iou = matches / total

    exact = sub_p == exp_p
    iou_delta = max(0.0, iou - prev_iou)
    step_reward = iou_delta - illegal_penalty - size_penalty

    return {
        "iou": iou,
        "iou_delta": iou_delta,
        "step_reward": step_reward,
        "illegal_count": illegal_count,
        "illegal_penalty": illegal_penalty,
        "size_mismatch": size_mismatch,
        "size_penalty": size_penalty,
        "expected_h": exp_h,
        "expected_w": exp_w,
        "submitted_h": sub_h,
        "submitted_w": sub_w,
        "exact": exact,
        "matches": matches,
        "total": total,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "sub_grid": sub,
        "sub_p": sub_p,
        "exp_p": exp_p,
        "parsed_text": _grid_to_text(sub),
    }


def score_side_view(submitted: str, expected: str) -> tuple[float, dict[str, Any]]:
    """Backward-compatible wrapper (single-step, no prior IoU)."""
    stats = evaluate_submission(submitted, expected, prev_iou=0.0)
    return stats["step_reward"], stats


class AsciiSideViewEnv(BaseEnv):
    """Multi-step refinement. Step reward = IoU gain − penalties. Default 7 steps × 7 episodes."""

    def __init__(self) -> None:
        self._rng = random.Random()
        self._puzzle: Puzzle | None = None
        self._seed = 0
        self._steps = 0
        self.max_steps = 7
        self._prev_iou = 0.0
        self._best_iou = 0.0
        self._episode_reward = 0.0
        self._last_parsed = ""

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._seed = 0 if seed is None else int(seed)
        self._rng.seed(self._seed)
        self.max_steps = min(35, int(params.get("max_steps", 7)))
        self._steps = 0
        self._prev_iou = 0.0
        self._best_iou = 0.0
        self._episode_reward = 0.0
        self._last_parsed = ""
        self._puzzle = self._rng.choice(CATALOG)
        return self._observation(include_feedback=False)

    def _observation(self, include_feedback: bool) -> dict[str, Any]:
        assert self._puzzle is not None
        obs: dict[str, Any] = {
            "task": "Given the front view and description, output the side view (90° rotation to the right).",
            "description": self._puzzle.description,
            "front_view": self._puzzle.front,
            "output_rules": (
                "STRICT OUTPUT FORMAT: reply with ONLY the side-view ASCII grid. "
                "Each line must contain ONLY # (solid) and . (empty) — no words, "
                "no explanation, no markdown fences, no labels. "
                f"Grid must be exactly {self._expected_shape()[0]} rows × "
                f"{self._expected_shape()[1]} columns (matching front-view canvas size)."
            ),
            "example": ".####.\n######\n.####.",
            "steps_left": self.max_steps - self._steps,
        }
        if include_feedback:
            obs["previous_iou"] = round(self._prev_iou, 4)
            obs["best_iou"] = round(self._best_iou, 4)
            obs["previous_submission"] = self._last_parsed
            obs["feedback"] = (
                f"Previous IoU={self._prev_iou:.3f}. "
                "Submit an improved side view (IoU gain − penalties per step)."
            )
        return obs

    def _expected_shape(self) -> tuple[int, int]:
        assert self._puzzle is not None
        return _expected_shape(self._puzzle.side)

    def step(self, action: Any) -> StepResult:
        if self._puzzle is None:
            raise RuntimeError("Call reset() before step()")

        self._steps += 1
        submitted = str(action)
        stats = evaluate_submission(submitted, self._puzzle.side, prev_iou=self._prev_iou)
        step_reward = stats["step_reward"]
        self._episode_reward += step_reward
        self._prev_iou = stats["iou"]
        self._best_iou = max(self._best_iou, stats["iou"])
        self._last_parsed = stats["parsed_text"]

        exact = stats["exact"]
        terminated = exact or self._steps >= self.max_steps
        truncated = self._steps >= self.max_steps and not exact

        if terminated or truncated:
            observation: dict[str, Any] = {
                "result": "done",
                "final_iou": round(stats["iou"], 4),
                "best_iou": round(self._best_iou, 4),
                "episode_reward": round(self._episode_reward, 4),
                "exact_match": exact,
            }
        else:
            observation = self._observation(include_feedback=True)

        return StepResult(
            observation=observation,
            reward=step_reward,
            terminated=terminated,
            truncated=truncated,
            info=self._info(submitted, stats),
        )

    def _info(self, submitted: str, stats: dict[str, Any]) -> dict[str, str]:
        assert self._puzzle is not None
        p = self._puzzle
        sub_p = stats["sub_p"]
        exp_p = stats["exp_p"]
        h, w = len(sub_p), len(sub_p[0]) if sub_p else 0
        diff_rows: list[str] = []
        for r in range(h):
            row: list[str] = []
            for c in range(w):
                if sub_p[r][c] == exp_p[r][c]:
                    row.append("=" if sub_p[r][c] == "#" else ".")
                elif sub_p[r][c] == "#":
                    row.append("+")
                else:
                    row.append("-")
            diff_rows.append("".join(row))

        return {
            "puzzle_name": p.name,
            "description": p.description,
            "front_view": p.front,
            "expected_side": p.side,
            "submitted_side": submitted,
            "parsed_side": stats["parsed_text"],
            "submitted_grid": json.dumps(sub_p),
            "expected_grid": json.dumps(exp_p),
            "diff_grid": json.dumps(diff_rows),
            "iou": f"{stats['iou']:.4f}",
            "final_iou": f"{stats['iou']:.4f}",
            "iou_delta": f"{stats['iou_delta']:.4f}",
            "illegal_count": str(stats["illegal_count"]),
            "illegal_penalty": f"{stats['illegal_penalty']:.4f}",
            "size_mismatch": "true" if stats["size_mismatch"] else "false",
            "size_penalty": f"{stats['size_penalty']:.4f}",
            "expected_shape": json.dumps([stats["expected_h"], stats["expected_w"]]),
            "submitted_shape": json.dumps([stats["submitted_h"], stats["submitted_w"]]),
            "exact_match": "true" if stats["exact"] else "false",
            "exact_numeric": "1" if stats["exact"] else "0",
            "matches": str(stats["matches"]),
            "total_cells": str(stats["total"]),
            "false_pos": str(stats["false_pos"]),
            "false_neg": str(stats["false_neg"]),
            "seed": str(self._seed),
            "steps_taken": str(self._steps),
            "episode_reward": f"{self._episode_reward:.4f}",
            "best_iou": f"{self._best_iou:.4f}",
        }
