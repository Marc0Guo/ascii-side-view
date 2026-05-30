"""Sync showcase/data/replay.js from replay.json (after mesocosm run export)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "data", "replay.json")
JS_PATH = os.path.join(HERE, "data", "replay.js")

with open(JSON_PATH) as f:
    data = json.load(f)
with open(JS_PATH, "w") as f:
    f.write("window.REPLAY = ")
    json.dump(data, f)
    f.write(";\n")
print(f"wrote {JS_PATH}")
