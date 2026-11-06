"""Regenerates every voiceover line with the TTS engine's own WordBoundary
events (ground truth, not ASR), converts each to WAV (removes MP3 encoder
priming-silence ambiguity), and writes assets/audio/vo/words.json -- the
one place word-level timestamps come from. Run from video/.
"""
import asyncio, json, subprocess, sys

sys.path.insert(0, "/Users/aviadmerlin/.local/share/uv/tools/edge-tts/lib/python3.10/site-packages")
import edge_tts

VOICE = "en-US-AndrewNeural"
RATE = "-5%"
OUT = "assets/audio/vo"

# 05_whatif split into two lines so the video cut lands in the gap BETWEEN
# lines (where I can size the gap deliberately) instead of mid-sentence.
LINES = [
    ("01_hook", "What if the 84th-minute goal never happened?"),
    ("02_problem_a", "We'd have won if that goal hadn't gone in."),
    ("03_problem_b", "Now you can test it."),
    ("04_turning_point", "Minute 84: one goal swings the match."),
    ("05a_cancel", "Cancel it."),
    ("05b_replay", "Replay the match."),
    ("06_reveal", "Without that goal, Dunmore Town's win chance jumps from 9% to 92%."),
    ("07_ask_agent", "Or just ask the AI agent."),
    ("08_answer", "Same answer, in plain English."),
    ("09_wordmark", "Sliding Doors."),
    ("10_tagline", "Stop debating the what-if. Simulate it."),
]


async def gen_one(line_id, text):
    comm = edge_tts.Communicate(text, VOICE, rate=RATE, boundary="WordBoundary")
    audio_path = f"{OUT}/{line_id}.mp3"
    words = []
    with open(audio_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({
                    "text": chunk["text"],
                    "start": round(chunk["offset"] / 1e7, 4),
                    "end": round((chunk["offset"] + chunk["duration"]) / 1e7, 4),
                })
    return words


def decompose_reveal_tail(words):
    """edge-tts merges "9% to 92%" into one WordBoundary token. Split it
    using word-level DURATIONS measured from isolated probes of the same
    engine/voice/rate ("nine percent" / "to" / "ninety two percent"),
    cross-validated against an isolated-tail probe of the full phrase
    (merged-span duration matched within 0.7-4.6%). This is the TTS
    engine's own timing, decomposed -- not ASR forced alignment, which
    measured ~1s off against the real WordBoundary data on this project's
    first pass (see video/README.md)."""
    FRAC_9PCT, FRAC_TO, FRAC_92PCT = 0.3809, 0.0952, 0.5238
    out = []
    for w in words:
        if w["text"] == "9% to 92%":
            start, end = w["start"], w["end"]
            dur = end - start
            t0, t1, t2, t3 = (
                start,
                start + FRAC_9PCT * dur,
                start + (FRAC_9PCT + FRAC_TO) * dur,
                end,
            )
            out.append({"text": "9%", "start": round(t0, 4), "end": round(t1, 4), "derived": "proportional-split"})
            out.append({"text": "to", "start": round(t1, 4), "end": round(t2, 4), "derived": "proportional-split"})
            out.append({"text": "92%", "start": round(t2, 4), "end": round(t3, 4), "derived": "proportional-split"})
        else:
            out.append(w)
    return out


TAIL_PAD = 0.12  # uniform trailing silence after the last word, all lines


async def main():
    result = {}
    for line_id, text in LINES:
        words = await gen_one(line_id, text)
        if line_id == "06_reveal":
            words = decompose_reveal_tail(words)

        # The service pads very short utterances with inconsistent trailing
        # silence (0.38s typical, up to 0.94s seen on 2-3 word lines) --
        # not a sync bug, just wasted scene time. Trim every line to the
        # same tail so scene durations are deterministic and pacing is
        # uniform.
        raw_dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", f"{OUT}/{line_id}.mp3"],
            capture_output=True, text=True).stdout.strip())
        last_word_end = words[-1]["end"]
        target_dur = min(raw_dur, last_word_end + TAIL_PAD)

        subprocess.run(["ffmpeg", "-y", "-i", f"{OUT}/{line_id}.mp3",
                         "-t", f"{target_dur:.4f}", "-ar", "48000", "-ac", "2",
                         f"{OUT}/{line_id}.wav"], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", f"{OUT}/{line_id}.mp3",
                         "-t", f"{target_dur:.4f}", "-c:a", "libmp3lame", "-q:a", "2",
                         f"{OUT}/{line_id}.trimmed.mp3"], capture_output=True)
        subprocess.run(["mv", f"{OUT}/{line_id}.trimmed.mp3", f"{OUT}/{line_id}.mp3"])

        result[line_id] = {"text": text, "duration": round(target_dur, 4),
                            "rawDuration": round(raw_dur, 4), "words": words}
        print(f"{line_id:16s} dur={target_dur:6.3f} (raw {raw_dur:6.3f})  "
              + " ".join(f"{w['text']}@{w['start']:.3f}" for w in words))
    json.dump(result, open(f"{OUT}/words.json", "w"), indent=2)
    print(f"\nwrote {OUT}/words.json")


asyncio.run(main())
