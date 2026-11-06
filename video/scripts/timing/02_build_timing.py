"""Builds assets/timing.json, the single source of truth for every scene
cut, text reveal and highlight in the film. Consumes assets/audio/vo/words.json
(real word-boundary data from the TTS engine -- see 01_generate_audio.py).

Nothing in here is a sync-critical hand-typed second: every absolute time
is computed from a real word timestamp plus a small set of named,
structural pacing constants (SCENES below) that are NOT audio-derived --
inter-line gaps, decorative pre-rolls, tail holds. Those are editorial
choices, not measurements, and are kept separate on purpose.

Run from video/. Writes assets/timing.json.
"""
import json

WORDS = json.load(open("assets/audio/vo/words.json"))

CAPTION_LEAD = 0.15   # text appears this long before its first spoken word
POST_SPEECH_MIN = 0.15  # minimum silence after a line before anything else may cut


def line(line_id):
    return WORDS[line_id]


def vo_start_local(line_id, cap_appear_local):
    """Audio start (scene-local) such that word 0 lands at cap_appear_local + CAPTION_LEAD."""
    w0 = line(line_id)["words"][0]["start"]
    return cap_appear_local + CAPTION_LEAD - w0


def vo_end_local(line_id, cap_appear_local):
    return vo_start_local(line_id, cap_appear_local) + line(line_id)["duration"]


def words_local(line_id, cap_appear_local):
    vs = vo_start_local(line_id, cap_appear_local)
    out = []
    for w in line(line_id)["words"]:
        out.append({**w, "startLocal": round(vs + w["start"], 4), "endLocal": round(vs + w["end"], 4)})
    return out


# ---------------------------------------------------------------------------
# Scene-by-scene layout. Each builder takes the scene's global start time and
# returns (sceneDuration, linesOut, cutsOut, sfxOut). Structural constants
# (gaps, holds, pre-rolls, SFX peak offsets) are named and commented as such.
# ---------------------------------------------------------------------------

SFX_PEAK = {  # measured via `hyperframes audio`, seconds from each file's own start to its peak
    "whoosh": 0.162,
    "click": 0.073,
    "tick": 0.073,
    "pop": 0.120,
    "impact": 1.986,
}


def build_hook(start):
    cap_appear_local = 0.0  # scene opens straight on the ghost-text hook, no pre-roll
    TAIL_HOLD = 0.25
    w = words_local("01_hook", cap_appear_local)
    duration = round(vo_end_local("01_hook", cap_appear_local) + TAIL_HOLD, 4)
    lines_out = {"01_hook": {
        "sceneId": "hook", "capAppearLocal": cap_appear_local,
        "voStartLocal": vo_start_local("01_hook", cap_appear_local),
        "voEndLocal": vo_end_local("01_hook", cap_appear_local),
        "voDuration": line("01_hook")["duration"], "words": w,
        "audioSrc": "assets/audio/vo/01_hook.wav",
    }}
    return duration, lines_out, {}, {}


def build_problem(start):
    INTER_LINE_GAP = 0.35  # beat for the strike-through before the answer lands
    TAIL_HOLD = 1.10  # "Now you can test it." needs ~2s readable after it lands (5 words)
    cap_a = 0.0
    w_a = words_local("02_problem_a", cap_a)
    end_a = vo_end_local("02_problem_a", cap_a)
    cap_b = round(end_a + INTER_LINE_GAP, 4)
    w_b = words_local("03_problem_b", cap_b)
    duration = round(vo_end_local("03_problem_b", cap_b) + TAIL_HOLD, 4)
    lines_out = {
        "02_problem_a": {"sceneId": "problem", "capAppearLocal": cap_a,
                          "voStartLocal": vo_start_local("02_problem_a", cap_a), "voEndLocal": end_a,
                          "voDuration": line("02_problem_a")["duration"], "words": w_a,
                          "audioSrc": "assets/audio/vo/02_problem_a.wav"},
        "03_problem_b": {"sceneId": "problem", "capAppearLocal": cap_b,
                          "voStartLocal": vo_start_local("03_problem_b", cap_b),
                          "voEndLocal": vo_end_local("03_problem_b", cap_b),
                          "voDuration": line("03_problem_b")["duration"], "words": w_b,
                          "audioSrc": "assets/audio/vo/03_problem_b.wav"},
    }
    cuts_out = {"problem_strike": {"sceneId": "problem", "timeLocal": round(end_a + 0.08, 4)}}
    return duration, lines_out, cuts_out, {}


def build_turning(start):
    TAIL_HOLD = 0.35
    cap_appear_local = 0.0
    w = words_local("04_turning_point", cap_appear_local)
    duration = round(vo_end_local("04_turning_point", cap_appear_local) + TAIL_HOLD, 4)
    swings_word = next(x for x in w if x["text"] == "swings")
    lines_out = {"04_turning_point": {
        "sceneId": "turning", "capAppearLocal": cap_appear_local,
        "voStartLocal": vo_start_local("04_turning_point", cap_appear_local),
        "voEndLocal": vo_end_local("04_turning_point", cap_appear_local),
        "voDuration": line("04_turning_point")["duration"], "words": w,
        "audioSrc": "assets/audio/vo/04_turning_point.wav",
    }}
    cuts_out = {"turning_ring": {"sceneId": "turning", "timeLocal": swings_word["startLocal"]}}
    sfx_out = {"sfx-pop-ring": {"sceneId": "turning",
                                 "startLocal": round(swings_word["startLocal"] - SFX_PEAK["pop"], 4),
                                 "volume": 0.30}}
    return duration, lines_out, cuts_out, sfx_out


def build_whatif(start):
    # Real captured-footage durations (fixed assets, not audio-derived):
    CLIP_A_VISUAL_NEED = 2.00   # assets/clips/whatif-a.mp4: dropdown open -> 84' selected
    CLIP_B_VISUAL_NEED = 1.10   # assets/clips/whatif-b.mp4: Run button -> click
    POST_CUT_PAD = 0.08
    TAIL_HOLD = 0.30
    # Real click frame inside clip B, measured from assets/clips/whatif.marks.json
    # (run_clicked 9.27 - media-start 8.85 = 0.42s into the clip).
    CLICK_IN_CLIP_B = 0.42

    cap_a = 0.0
    w_a = words_local("05a_cancel", cap_a)
    end_a = vo_end_local("05a_cancel", cap_a)
    cut_time = round(max(CLIP_A_VISUAL_NEED, end_a + POST_SPEECH_MIN), 4)

    cap_b = round(cut_time + POST_CUT_PAD, 4)
    w_b = words_local("05b_replay", cap_b)
    end_b = vo_end_local("05b_replay", cap_b)
    clip_b_duration = round(max(CLIP_B_VISUAL_NEED, end_b + TAIL_HOLD - cut_time), 4)
    duration = round(cut_time + clip_b_duration, 4)

    lines_out = {
        "05a_cancel": {"sceneId": "whatif", "capAppearLocal": cap_a,
                       "voStartLocal": vo_start_local("05a_cancel", cap_a), "voEndLocal": end_a,
                       "voDuration": line("05a_cancel")["duration"], "words": w_a,
                       "audioSrc": "assets/audio/vo/05a_cancel.wav"},
        "05b_replay": {"sceneId": "whatif", "capAppearLocal": cap_b,
                       "voStartLocal": vo_start_local("05b_replay", cap_b), "voEndLocal": end_b,
                       "voDuration": line("05b_replay")["duration"], "words": w_b,
                       "audioSrc": "assets/audio/vo/05b_replay.wav"},
    }
    cuts_out = {
        "whatif_cut": {"sceneId": "whatif", "timeLocal": cut_time,
                        "note": "gap between 05a end (%.3f) and 05b start -- silence margin %.3fs"
                                % (end_a, cut_time - end_a)},
        "whatif_click": {"sceneId": "whatif", "timeLocal": round(cut_time + CLICK_IN_CLIP_B, 4)},
    }
    sfx_out = {
        "sfx-whoosh-whatif": {"sceneId": "whatif", "startLocal": round(cut_time - SFX_PEAK["whoosh"], 4), "volume": 0.30},
        "sfx-click-run": {"sceneId": "whatif",
                           "startLocal": round(cut_time + CLICK_IN_CLIP_B - SFX_PEAK["click"], 4), "volume": 0.40},
    }
    return duration, lines_out, cuts_out, sfx_out, cut_time, clip_b_duration


def build_reveal(start):
    TAIL_HOLD = 1.1
    cap_appear_local = 0.0
    w = words_local("06_reveal", cap_appear_local)
    duration = round(vo_end_local("06_reveal", cap_appear_local) + TAIL_HOLD, 4)
    w9 = next(x for x in w if x["text"] == "9%")
    wto = next(x for x in w if x["text"] == "to")
    w92 = next(x for x in w if x["text"] == "92%")
    lines_out = {"06_reveal": {
        "sceneId": "reveal", "capAppearLocal": cap_appear_local,
        "voStartLocal": vo_start_local("06_reveal", cap_appear_local),
        "voEndLocal": vo_end_local("06_reveal", cap_appear_local),
        "voDuration": line("06_reveal")["duration"], "words": w,
        "audioSrc": "assets/audio/vo/06_reveal.wav",
    }}
    cuts_out = {
        "reveal_9pct": {"sceneId": "reveal", "timeLocal": w9["startLocal"]},
        "reveal_to": {"sceneId": "reveal", "timeLocal": wto["startLocal"]},
        "reveal_92pct": {"sceneId": "reveal", "timeLocal": w92["startLocal"]},
    }
    sfx_out = {"sfx-impact-reveal": {"sceneId": "reveal",
                                      "startLocal": round(w92["startLocal"] - SFX_PEAK["impact"], 4),
                                      "volume": 0.22}}
    proof_at = round(w92["startLocal"] + 0.31, 4)
    cuts_out["reveal_proof"] = {"sceneId": "reveal", "timeLocal": proof_at}
    return duration, lines_out, cuts_out, sfx_out


def build_agent(start):
    TYPE_CLIP = 2.2   # real captured+sped-up typing footage (fixed asset)
    THINK_CLIP = 0.9  # real captured thinking-spinner footage (fixed asset)
    ANSWER_MIN = 2.6  # minimum time to hold the answer screenshot
    POST_ANSWER_PAD = 0.08
    TAIL_HOLD = 0.30
    # Real click frame inside the sped-up type clip, derived from
    # assets/clips/agent.marks.json: (ask_clicked 9.05 - media_start 5.6) / speed 1.6.
    ASK_CLICK_IN_CLIP = round((9.05 - 5.6) / 1.6, 4)

    cap_7 = 0.0
    w7 = words_local("07_ask_agent", cap_7)
    end_7 = vo_end_local("07_ask_agent", cap_7)
    cut_1 = TYPE_CLIP               # type -> think
    cut_2 = TYPE_CLIP + THINK_CLIP  # think -> answer
    assert cut_1 > end_7 + POST_SPEECH_MIN, "line07 speech runs into the type/think cut"

    cap_8 = round(cut_2 + POST_ANSWER_PAD, 4)
    w8 = words_local("08_answer", cap_8)
    end_8 = vo_end_local("08_answer", cap_8)
    answer_duration = round(max(ANSWER_MIN, end_8 + TAIL_HOLD - cut_2), 4)
    duration = round(cut_2 + answer_duration, 4)

    lines_out = {
        "07_ask_agent": {"sceneId": "agent", "capAppearLocal": cap_7,
                          "voStartLocal": vo_start_local("07_ask_agent", cap_7), "voEndLocal": end_7,
                          "voDuration": line("07_ask_agent")["duration"], "words": w7,
                          "audioSrc": "assets/audio/vo/07_ask_agent.wav"},
        "08_answer": {"sceneId": "agent", "capAppearLocal": cap_8,
                      "voStartLocal": vo_start_local("08_answer", cap_8), "voEndLocal": end_8,
                      "voDuration": line("08_answer")["duration"], "words": w8,
                      "audioSrc": "assets/audio/vo/08_answer.wav"},
    }
    cuts_out = {
        "agent_cut_type_think": {"sceneId": "agent", "timeLocal": cut_1},
        "agent_cut_answer": {"sceneId": "agent", "timeLocal": cut_2},
        "agent_ask_click": {"sceneId": "agent", "timeLocal": round(ASK_CLICK_IN_CLIP, 4)},
    }
    sfx_out = {
        "sfx-whoosh-agent": {"sceneId": "agent", "startLocal": round(cut_1 - SFX_PEAK["whoosh"], 4), "volume": 0.30},
        "sfx-click-ask": {"sceneId": "agent",
                           "startLocal": round(ASK_CLICK_IN_CLIP - SFX_PEAK["click"], 4), "volume": 0.40},
        "sfx-tick-thinking": {"sceneId": "agent",
                               "startLocal": round(cut_1 + 0.15 - SFX_PEAK["tick"], 4), "volume": 0.22},
    }
    return duration, lines_out, cuts_out, sfx_out, cut_1, cut_2, answer_duration


def build_closing(start):
    SEAM_LEAD_IN = 0.35    # decorative pre-roll before the wordmark slides out
    INTER_LINE_GAP = 0.30
    TAIL_HOLD = 4.6         # silent credits hold -- the 17-word hackathon line alone needs ~4.2s readable

    cap_9 = SEAM_LEAD_IN
    w9 = words_local("09_wordmark", cap_9)
    end_9 = vo_end_local("09_wordmark", cap_9)
    cap_10 = round(end_9 + INTER_LINE_GAP, 4)
    w10 = words_local("10_tagline", cap_10)
    end_10 = vo_end_local("10_tagline", cap_10)
    duration = round(end_10 + TAIL_HOLD, 4)

    lines_out = {
        "09_wordmark": {"sceneId": "closing", "capAppearLocal": cap_9,
                         "voStartLocal": vo_start_local("09_wordmark", cap_9), "voEndLocal": end_9,
                         "voDuration": line("09_wordmark")["duration"], "words": w9,
                         "audioSrc": "assets/audio/vo/09_wordmark.wav"},
        "10_tagline": {"sceneId": "closing", "capAppearLocal": cap_10,
                       "voStartLocal": vo_start_local("10_tagline", cap_10), "voEndLocal": end_10,
                       "voDuration": line("10_tagline")["duration"], "words": w10,
                       "audioSrc": "assets/audio/vo/10_tagline.wav"},
    }
    cuts_out = {
        "closing_seam": {"sceneId": "closing", "timeLocal": 0.0},
        "closing_credits": {"sceneId": "closing", "timeLocal": round(end_10 + 0.3, 4)},
    }
    return duration, lines_out, cuts_out, {}


def main():
    scenes, lines_out, cuts_out, sfx_out = {}, {}, {}, {}
    t = 0.0

    for scene_id, builder in [("hook", build_hook), ("problem", build_problem),
                               ("turning", build_turning)]:
        dur, lns, cts, sfx = builder(t)
        scenes[scene_id] = {"start": round(t, 4), "duration": dur}
        for k, v in lns.items():
            v["voStart"] = round(t + v["voStartLocal"], 4)
            v["voEnd"] = round(t + v["voEndLocal"], 4)
            v["capAppear"] = round(t + v["capAppearLocal"], 4)
            for w in v["words"]:
                w["start"], w["end"] = w.pop("start"), w.pop("end")  # keep local-relative keys authoritative below
                w["startAbs"] = round(t + w["startLocal"], 4)
                w["endAbs"] = round(t + w["endLocal"], 4)
            lines_out[k] = v
        for k, v in cts.items():
            cuts_out[k] = {**v, "timeAbs": round(t + v["timeLocal"], 4)}
        for k, v in sfx.items():
            sfx_out[k] = {**v, "start": round(t + v.pop("startLocal"), 4)}
        t += dur

    dur, lns, cts, sfx, cutA, clipBDur = build_whatif(t)
    scenes["whatif"] = {"start": round(t, 4), "duration": dur}
    for k, v in lns.items():
        v["voStart"] = round(t + v["voStartLocal"], 4)
        v["voEnd"] = round(t + v["voEndLocal"], 4)
        v["capAppear"] = round(t + v["capAppearLocal"], 4)
        for w in v["words"]:
            w["startAbs"] = round(t + w["startLocal"], 4)
            w["endAbs"] = round(t + w["endLocal"], 4)
        lines_out[k] = v
    for k, v in cts.items():
        cuts_out[k] = {**v, "timeAbs": round(t + v["timeLocal"], 4)}
    for k, v in sfx.items():
        sfx_out[k] = {**v, "start": round(t + v.pop("startLocal"), 4)}
    whatif_clip_durations = {"whatif-a": cutA, "whatif-b": clipBDur}
    t += dur

    dur, lns, cts, sfx = build_reveal(t)
    scenes["reveal"] = {"start": round(t, 4), "duration": dur}
    for k, v in lns.items():
        v["voStart"] = round(t + v["voStartLocal"], 4)
        v["voEnd"] = round(t + v["voEndLocal"], 4)
        v["capAppear"] = round(t + v["capAppearLocal"], 4)
        for w in v["words"]:
            w["startAbs"] = round(t + w["startLocal"], 4)
            w["endAbs"] = round(t + w["endLocal"], 4)
        lines_out[k] = v
    for k, v in cts.items():
        cuts_out[k] = {**v, "timeAbs": round(t + v["timeLocal"], 4)}
    for k, v in sfx.items():
        sfx_out[k] = {**v, "start": round(t + v.pop("startLocal"), 4)}
    t += dur

    dur, lns, cts, sfx, cut1, cut2, answerDur = build_agent(t)
    scenes["agent"] = {"start": round(t, 4), "duration": dur}
    for k, v in lns.items():
        v["voStart"] = round(t + v["voStartLocal"], 4)
        v["voEnd"] = round(t + v["voEndLocal"], 4)
        v["capAppear"] = round(t + v["capAppearLocal"], 4)
        for w in v["words"]:
            w["startAbs"] = round(t + w["startLocal"], 4)
            w["endAbs"] = round(t + w["endLocal"], 4)
        lines_out[k] = v
    for k, v in cts.items():
        cuts_out[k] = {**v, "timeAbs": round(t + v["timeLocal"], 4)}
    for k, v in sfx.items():
        sfx_out[k] = {**v, "start": round(t + v.pop("startLocal"), 4)}
    agent_answer_duration = answerDur
    t += dur

    dur, lns, cts, sfx = build_closing(t)
    scenes["closing"] = {"start": round(t, 4), "duration": dur}
    for k, v in lns.items():
        v["voStart"] = round(t + v["voStartLocal"], 4)
        v["voEnd"] = round(t + v["voEndLocal"], 4)
        v["capAppear"] = round(t + v["capAppearLocal"], 4)
        for w in v["words"]:
            w["startAbs"] = round(t + w["startLocal"], 4)
            w["endAbs"] = round(t + w["endLocal"], 4)
        lines_out[k] = v
    for k, v in cts.items():
        cuts_out[k] = {**v, "timeAbs": round(t + v["timeLocal"], 4)}
    t += dur

    total = round(t, 4)
    manifest = {
        "fps": 30,
        "totalDuration": total,
        "scenes": scenes,
        "lines": lines_out,
        "cuts": cuts_out,
        "sfx": sfx_out,
        "clipDurations": {
            "whatif-a": whatif_clip_durations["whatif-a"],
            "whatif-b": whatif_clip_durations["whatif-b"],
            "agent-type": 2.2, "agent-think": 0.9,
            "agent-answer": agent_answer_duration,
        },
    }
    json.dump(manifest, open("assets/timing.json", "w"), indent=2)

    print(f"TOTAL DURATION: {total}s (cap 38s)\n")
    for sid, s in scenes.items():
        print(f"{sid:10s} start={s['start']:7.3f}  duration={s['duration']:6.3f}  end={s['start']+s['duration']:7.3f}")
    print()
    for lid, l in lines_out.items():
        print(f"{lid:16s} scene={l['sceneId']:9s} capAppear={l['capAppear']:7.3f}  voStart={l['voStart']:7.3f}  voEnd={l['voEnd']:7.3f}")
    print()
    for cid, c in cuts_out.items():
        print(f"cut {cid:24s} scene={c['sceneId']:9s} local={c['timeLocal']:7.3f}  abs={c['timeAbs']:7.3f}")
    print()
    for sid, s in sfx_out.items():
        print(f"sfx {sid:20s} scene={s['sceneId']:9s} abs_start={s['start']:7.3f}")


if __name__ == "__main__":
    main()
