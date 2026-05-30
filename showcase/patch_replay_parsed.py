"""Backfill parsed_side in an exported replay.json using current env parser."""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import _grid_to_text, _parse_ascii, _trim_grid  # noqa: E402

REPLAY = os.path.join(os.path.dirname(__file__), "data", "replay.json")


def main() -> None:
    with open(REPLAY) as f:
        data = json.load(f)
    n = 0
    for turns in data.get("replay", {}).values():
        for turn in turns:
            info = turn.get("info") or {}
            raw = info.get("submitted_side") or turn.get("action") or ""
            grid = _trim_grid(_parse_ascii(raw))
            info["parsed_side"] = _grid_to_text(grid)
            turn["info"] = info
            n += 1
    with open(REPLAY, "w") as f:
        json.dump(data, f, indent=2)
    print(f"patched parsed_side on {n} turns → {REPLAY}")


if __name__ == "__main__":
    main()
