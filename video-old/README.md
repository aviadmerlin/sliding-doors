# Sliding Doors — hackathon demo video

~1:32 demo video for the Microsoft x Premier League "Inside the Game"
hackathon submission, built with [Remotion](https://remotion.dev). Composed
from real recorded footage of the live app (no fabricated UI), an Azure AI
Speech voiceover, and no music (per the hackathon rules: no third-party
trademarks and no copyrighted audio).

## Pipeline

1. **Record real footage** of the live app with Playwright:

   ```bash
   npm i
   npx playwright install chromium
   cp .env.example .env   # fill in AZURE_SPEECH_KEY / AZURE_SPEECH_REGION / APP_URL
   node scripts/record-footage.mjs
   ```

   Produces `public/recordings/desktop.webm`, `public/recordings/mobile.webm`,
   and `public/recordings/timestamps.json` (ms-since-start markers for every
   beat: scoreboard, turning points, the what-if counterfactual run, each
   agent question/answer, the club profile cards). Gitignored — rerun to
   regenerate; the live app's real output may vary slightly run to run
   (the counterfactual re-simulates with fresh randomness each time).

2. **Generate the voiceover** with Azure AI Speech (reuses the
   `sliding-doors-resource` Cognitive Services account, `switzerlandnorth`,
   same resource as the agent's Azure OpenAI deployment — no new Azure
   resource needed):

   ```bash
   node scripts/generate-voiceover.mjs
   ```

   Produces `public/audio/01-hook.mp3` .. `06-outro.mp3`. ~1,100 characters
   total, so this costs about $0.02 at the standard neural-voice rate
   ($16/1M characters).

3. **Edit `src/marks.ts`** if you re-recorded footage — the scene
   components (`src/scenes/*.tsx`) cut specific real moments out of
   `desktop.webm`/`mobile.webm` by timestamp (e.g. the exact frame the
   counterfactual result or an agent answer appears), so new timestamps
   need copying in from the new `public/recordings/timestamps.json`.

4. **Preview**: `npm run dev` (opens Remotion Studio).

5. **Render**: `npx remotion render SlidingDoorsDemo out/sliding-doors-demo.mp4 --codec=h264`

## Structure

```
scripts/
  record-footage.mjs      # Playwright: records the live app, writes timestamps.json
  generate-voiceover.mjs  # Azure AI Speech: narration per scene
src/
  marks.ts                # recorded timestamps, copied from public/recordings/timestamps.json
  Clip.tsx                # one trimmed cut from a recorded source file
  Caption.tsx             # lower-third caption bar
  VoiceOver.tsx           # per-scene narration audio
  SlidingDoorsDemo.tsx    # assembles the 6 scenes with crossfade transitions
  scenes/                 # one component per scene (hook, turning points,
                           # what-if, agent Q&A, calibration, outro)
public/
  calibration.png         # the one static (non-live-app) asset: the model's
                           # out-of-sample calibration chart, from assets/
                           # at the repo root -- framed as an "under the
                           # hood" validation insert, not the whole video
  recordings/, audio/     # gitignored -- regenerate via the scripts above
```

## Why some of scene 3 and 4 are sped up or cut

The counterfactual engine takes ~7s to re-simulate hundreds of matches, and
the AI agent takes ~16-20s per real answer (gpt-5-mini reasoning latency).
Scene 3 plays that computation sped up (`playbackRate`) rather than cut
entirely, since that wait *is* the proof it's really computing. Scene 4
hard-cuts from "question asked" to "answer appears" for each of the three
questions (English, Hebrew, and the honest refusal) rather than sitting
through three real ~20s waits — still real recorded frames before and
after, just not the dead time in between.
