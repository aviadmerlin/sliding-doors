"""Regenerates index.html from assets/timing.json. Scene hosts, VO/SFX
<audio> tags and the root duration are all written here, mechanically,
from the manifest -- never hand-typed. The manifest is also embedded
verbatim as window.__TIMING__ so every composition's script can read its
own word/cut times at runtime instead of containing literal seconds.

Run from video/. Rewrites index.html.
"""
import json

T = json.load(open("assets/timing.json"))
S = T["scenes"]

SCENE_FILES = {
    "hook": "hook", "problem": "problem", "turning": "turning",
    "whatif": "whatif", "reveal": "reveal", "agent": "agent", "closing": "closing",
}

scene_divs = []
for i, (sid, file) in enumerate(SCENE_FILES.items(), start=1):
    sc = S[sid]
    scene_divs.append(f"""      <div
        id="{sid}"
        data-composition-id="{sid}"
        data-composition-src="compositions/{file}.html"
        data-start="{sc['start']}"
        data-duration="{sc['duration']}"
        data-track-index="1"
        data-track-kind="graphics"
        data-width="1920"
        data-height="1080"
        style="z-index: {i}"
      ></div>""")

vo_tags = []
for lid, l in T["lines"].items():
    vo_tags.append(
        f'      <audio id="vo-{lid}" src="{l["audioSrc"]}" data-start="{l["voStart"]}" data-volume="1"></audio>'
    )

sfx_tags = []
for sid, s in T["sfx"].items():
    sfx_tags.append(
        f'      <audio id="{sid}" src="assets/audio/sfx/{sid.split("sfx-")[1].rsplit("-", 1)[0] if False else SFX_FILE[sid]}.mp3" '
        f'data-start="{s["start"]}" data-volume="{s["volume"]}"></audio>'
    ) if False else None

SFX_FILE = {
    "sfx-pop-ring": "pop",
    "sfx-whoosh-whatif": "whoosh",
    "sfx-click-run": "click",
    "sfx-impact-reveal": "impact",
    "sfx-whoosh-agent": "whoosh",
    "sfx-click-ask": "click",
    "sfx-tick-thinking": "tick",
}
sfx_tags = []
for sid, s in T["sfx"].items():
    sfx_tags.append(
        f'      <audio id="{sid}" src="assets/audio/sfx/{SFX_FILE[sid]}.mp3" '
        f'data-start="{s["start"]}" data-volume="{s["volume"]}"></audio>'
    )

timing_json_inline = json.dumps(T, separators=(",", ":"))

html = f"""<!doctype html>
<html
  lang="en"
  data-composition-variables='[
    {{"id":"brandBg","type":"color","label":"Background","default":"#0B1220"}},
    {{"id":"brandGreen","type":"color","label":"Up / after","default":"#22C55E"}},
    {{"id":"brandRed","type":"color","label":"Down / real","default":"#EF4444"}},
    {{"id":"brandAmber","type":"color","label":"Fork / marker","default":"#F59E0B"}},
    {{"id":"creditName","type":"string","label":"Credit name","default":"Aviad Merlin"}},
    {{"id":"creditUrl","type":"string","label":"URL line (optional)","default":""}}
  ]'
>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>
    <style>
      :root {{
        --bg: #0b1220;
        --green: #22c55e;
        --red: #ef4444;
        --amber: #f59e0b;
        --ink: #e5e7eb;
        --muted: #9ca3af;
        --line: #243049;
      }}
      * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }}
      html,
      body {{
        margin: 0;
        width: 1920px;
        height: 1080px;
        overflow: hidden;
        background: var(--bg);
        font-family: Inter, sans-serif;
      }}
      /* Untimed full-bleed base so no cut ever flashes through to white. */
      #stage-bg {{
        position: absolute;
        inset: 0;
        background: var(--bg);
        z-index: 0;
      }}
    </style>
  </head>
  <body>
    <!-- GENERATED FILE -- do not hand-edit.
         Source of truth: assets/timing.json
         Regenerate: python3 scripts/timing/03_apply_timing.py
         (which itself is built from assets/audio/vo/words.json by
         scripts/timing/02_build_timing.py -- see video/README.md) -->
    <script>
      // The single timing manifest. Every composition reads its own
      // scene/line/cut/sfx times from this object at runtime -- no
      // composition file contains a literal sync-critical second.
      window.__TIMING__ = {timing_json_inline};
    </script>

    <div
      id="root"
      data-composition-id="main"
      data-start="0"
      data-duration="{T['totalDuration']}"
      data-width="1920"
      data-height="1080"
    >
      <div id="stage-bg"></div>

      <!-- ── Scenes ─────────────────────────────────────────────── -->
{chr(10).join(scene_divs)}

      <!-- ── Voiceover (edge-tts en-US-AndrewNeural, rate -5%) ────
           WAV source (not MP3): removes LAME encoder priming-silence
           ambiguity at the exact sample HyperFrames seeks to. Word-level
           times are in window.__TIMING__.lines[id].words, from the TTS
           engine's own WordBoundary events (assets/audio/vo/words.json),
           not ASR. -->
{chr(10).join(vo_tags)}

      <!-- ── Sound effects ───────────────────────────────────────
           Each start = (cut or word time) - (that file's measured peak
           offset, from `hyperframes audio`), so the peak lands on the
           picture. Times computed in assets/timing.json. -->
{chr(10).join(sfx_tags)}

      <!-- ══ MUSIC SLOT ═══════════════════════════════════════════
           Drop a track at assets/audio/music.mp3, then remove this
           comment wrapper so the <audio> below goes live. Re-run
           scripts/timing/03_apply_timing.py first if timing.json has
           changed since -- the duck points below are written by hand
           against the CURRENT manifest's scene boundaries and need
           re-checking against window.__TIMING__ if durations moved.
      <audio
        id="music"
        src="assets/audio/music.mp3"
        data-start="0"
        data-duration="{T['totalDuration']}"
        data-volume="0.5"
        data-fade-in="1.2"
        data-fade-out="2.5"
      ></audio>
      ═══════════════════════════════════════════════════════════ -->
    </div>

    <script>
      window.__timelines = window.__timelines || {{}};
    </script>
  </body>
</html>
"""

open("index.html", "w").write(html)
print(f"wrote index.html  (total {T['totalDuration']}s)")
