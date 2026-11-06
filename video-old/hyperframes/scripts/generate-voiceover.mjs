// Generates narration for the HyperFrames rebuild with Azure AI Speech
// (same engine/voice as both prior versions, en-US-AndrewNeural), using
// SSML for per-line tuning and capturing the SDK's native wordBoundary
// events for word-synced captions / kinetic-type timing.
//
// Writes, per beat, to assets/audio/:
//   <id>.mp3          -- the narration audio
//   <id>.words.json   -- [{id, text, start, end}] in SECONDS, matching
//                        the word format hyperframes captions.md expects
//
// Usage: node scripts/generate-voiceover.mjs [beatId ...]
//   (no args = all beats; otherwise only the given beat ids)
import sdk from "microsoft-cognitiveservices-speech-sdk";
import { mkdirSync, writeFileSync } from "fs";
import "dotenv/config";

const KEY = process.env.AZURE_SPEECH_KEY;
const REGION = process.env.AZURE_SPEECH_REGION;
if (!KEY || !REGION) {
  throw new Error("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in video/.env");
}

const VOICE = "en-US-AndrewNeural";
const OUT_DIR = new URL("../assets/audio/", import.meta.url).pathname;
mkdirSync(OUT_DIR, { recursive: true });

const ssmlWrap = (inner) => `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
  <voice name="${VOICE}">${inner}</voice>
</speak>`;

// Beat ids and lines match video/SCRIPT.md exactly (approved 2026-11-02).
const beats = [
  {
    id: "01-hook",
    text: "What if that goal never happened?",
    ssml: ssmlWrap(`<prosody rate="-3%">What if that goal never happened?</prosody>`),
  },
  {
    id: "02-tension",
    text: "It's always just opinion. Two fans, same goal, two different stories.",
    ssml: ssmlWrap(
      `It's always just opinion. <break time="250ms"/> Two fans, same goal, two different stories.`,
    ),
  },
  {
    id: "03-reveal",
    text: "Until now. This is Sliding Doors.",
    ssml: ssmlWrap(`Until now. <break time="300ms"/> <prosody rate="-5%">This is Sliding Doors.</prosody>`),
  },
  {
    id: "04-proof-1",
    text: "Oakhaven were losing for forty minutes. Pick the moment that changed it.",
    ssml: ssmlWrap(
      `Oakhaven were losing for forty minutes. <break time="200ms"/> Pick the moment that changed it.`,
    ),
  },
  {
    id: "04-proof-2",
    text: "Cancel the goal — the engine re-simulates the match, 3,600 times.",
    ssml: ssmlWrap(
      `Cancel the goal <break time="150ms"/> the engine re-simulates the match, 3,600 times.`,
    ),
  },
  {
    id: "04-proof-3",
    text: "Or just ask — in English,",
    ssml: ssmlWrap(`Or just ask <break time="150ms"/> in English,`),
  },
  {
    id: "04-proof-4",
    text: "or Hebrew.",
    ssml: ssmlWrap(`or Hebrew.`),
  },
  {
    id: "05-payoff",
    text: "Dunmore's chances of winning: nine percent... to ninety-two.",
    ssml: ssmlWrap(
      `Dunmore's chances of winning: nine percent <break time="500ms"/> <prosody pitch="+8%">to ninety-two.</prosody>`,
    ),
  },
  {
    id: "06-claim",
    text: "Two agents, on Microsoft Foundry. Every number comes from the model, never made up.",
    ssml: ssmlWrap(
      `Two agents, on Microsoft Foundry. <break time="200ms"/> Every number comes from the model, never made up.`,
    ),
  },
  {
    id: "07-end-card",
    text: "Sliding Doors. Every match, every moment, every what if.",
    ssml: ssmlWrap(
      `<prosody rate="-8%">Sliding Doors. <break time="400ms"/> Every match, every moment, every what if.</prosody>`,
    ),
  },
];

function synthesize(ssml, outPath) {
  return new Promise((resolve, reject) => {
    const speechConfig = sdk.SpeechConfig.fromSubscription(KEY, REGION);
    speechConfig.speechSynthesisVoiceName = VOICE;
    speechConfig.speechSynthesisOutputFormat =
      sdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3;
    const audioConfig = sdk.AudioConfig.fromAudioFileOutput(outPath);
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig);

    const words = [];
    synthesizer.wordBoundary = (_sender, event) => {
      if (event.boundaryType !== "WordBoundary") return;
      const startMs = event.audioOffset / 10000;
      const durationMs = event.duration / 10000;
      // SDK quirk (found building v1): a word immediately followed by an
      // SSML <break> sometimes reports .text as that word plus the rest
      // of the sentence appended. Timing stays correct; sanitize to the
      // first token.
      const text = event.text.trim().split(/\s+/)[0];
      words.push({ text, startMs, endMs: startMs + durationMs });
    };

    synthesizer.speakSsmlAsync(
      ssml,
      (result) => {
        synthesizer.close();
        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          resolve(words);
        } else {
          reject(new Error(`Synthesis failed: ${result.errorDetails}`));
        }
      },
      (err) => {
        synthesizer.close();
        reject(err);
      },
    );
  });
}

const requested = process.argv.slice(2);
const toRun = requested.length ? beats.filter((b) => requested.includes(b.id)) : beats;
if (requested.length && toRun.length !== requested.length) {
  const found = toRun.map((b) => b.id);
  throw new Error(`Unknown beat id(s): ${requested.filter((r) => !found.includes(r))}`);
}

let totalChars = 0;
for (const beat of toRun) {
  const mp3Path = `${OUT_DIR}${beat.id}.mp3`;
  console.log(`Synthesizing ${beat.id} (${beat.text.length} chars)...`);
  const words = await synthesize(beat.ssml, mp3Path);

  // HyperFrames captions.md word format: {id, text, start, end} in SECONDS.
  const wordRecords = words.map((w, i) => ({
    id: `w${i}`,
    text: w.text,
    start: Math.round(w.startMs) / 1000,
    end: Math.round(w.endMs) / 1000,
  }));
  writeFileSync(`${OUT_DIR}${beat.id}.words.json`, JSON.stringify(wordRecords, null, 2));
  console.log(`  -> ${words.length} words, last ends at ${Math.round(words.at(-1).endMs)}ms`);
  totalChars += beat.text.length;
}
console.log(`Done. Characters synthesized this run: ${totalChars} (~$${((totalChars / 1_000_000) * 16).toFixed(3)})`);
