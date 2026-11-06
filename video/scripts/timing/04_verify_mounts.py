"""Cross-checks every composition's literal data-duration="..." mount
attribute against assets/timing.json's scene durations. Catches the exact
class of bug where a scene's hold time grows (edit SCENES in
02_build_timing.py, re-run 02+03) but a hand-authored mount window in a
composition file is left pointing at the old number.

Run from video/ after any timing.json rebuild, before `hyperframes check`.
"""
import json
import re
from pathlib import Path

T = json.load(open("assets/timing.json"))

# Elements that are authored to mount for the *entire* scene (data-start=0)
# and so must carry that scene's exact current duration.
WHOLE_SCENE_IDS = {
    "hook.html": ["hk-l1", "hk-l2", "hk-l3"],
    "problem.html": ["pb-quote", "pb-answer"],
    "turning.html": ["tn-score", "tn-stage", "tn-chart", "tn-cap"],
    "reveal.html": ["rv-kicker", "rv-left", "rv-right", "rv-proof-tag", "rv-proof"],
    "closing.html": ["cl-wordmark", "cl-tagline", "cl-credits"],
}
SCENE_OF_FILE = {
    "hook.html": "hook", "problem.html": "problem", "turning.html": "turning",
    "whatif.html": "whatif", "reveal.html": "reveal", "agent.html": "agent", "closing.html": "closing",
}

ok = True
for fname, ids in WHOLE_SCENE_IDS.items():
    path = Path("compositions") / fname
    html = path.read_text()
    expected = T["scenes"][SCENE_OF_FILE[fname]]["duration"]
    for elid in ids:
        m = re.search(rf'id="{elid}"[^>]*data-duration="([\d.]+)"', html) or \
            re.search(rf'data-duration="([\d.]+)"[^>]*id="{elid}"', html)
        if not m:
            print(f"SKIP  {fname}#{elid}: no data-duration found near this id (check manually)")
            continue
        got = float(m.group(1))
        if abs(got - expected) > 0.001:
            print(f"STALE {fname}#{elid}: data-duration={got} but window.__TIMING__.scenes.{SCENE_OF_FILE[fname]}.duration={expected}")
            ok = False
        else:
            print(f"ok    {fname}#{elid}: {got}")

print("\nALL MOUNTS MATCH THE MANIFEST" if ok else "\nFIX THE STALE MOUNTS ABOVE BEFORE RENDERING")
raise SystemExit(0 if ok else 1)
