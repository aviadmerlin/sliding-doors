// Generates narration for the v2 video with Azure AI Speech (same engine/
// voice as the first video, en-US-AndrewNeural), using SSML for per-line
// rate/pitch/pause tuning (see video/SCRIPT.md "Delivery notes") and
// capturing the SDK's native wordBoundary events for karaoke captions.
//
// Writes, per scene, to public/audio-v2/:
//   <id>.mp3            -- the narration audio
//   <id>.captions.json   -- Caption[] (word-level, @remotion/captions shape)
//
// Usage: node scripts/generate-voiceover-v2.mjs [sceneId ...]
//   (no args = all scenes; otherwise only the given scene ids)
import sdk from "microsoft-cognitiveservices-speech-sdk";
import { mkdirSync, writeFileSync } from "fs";
import "dotenv/config";

const KEY = process.env.AZURE_SPEECH_KEY;
const REGION = process.env.AZURE_SPEECH_REGION;
if (!KEY || !REGION) {
  throw new Error("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in video/.env");
}

const VOICE = "en-US-AndrewNeural";
const OUT_DIR = new URL("../public/audio-v2/", import.meta.url).pathname;
mkdirSync(OUT_DIR, { recursive: true });

const ssmlWrap = (inner) => `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
  <voice name="${VOICE}">${inner}</voice>
</speak>`;

// Plain text (for the captions.json "full line" record) must match what's
// actually spoken, ignoring SSML tags/breaks.
const scenes = [
  {
    id: "01-hook",
    text: "Every fan's had this argument — if that shot had gone in, we'd have won.",
    ssml: ssmlWrap(
      `<prosody rate="-5%">Every fan's had this argument <break time="250ms"/> if that shot had gone in, we'd have won.</prosody>`,
    ),
  },
  {
    id: "02-problem",
    text: "It's always just opinion. Nobody's ever been able to prove it — until now.",
    ssml: ssmlWrap(
      `It's always just opinion. <break time="300ms"/> Nobody's ever been able to prove it <break time="200ms"/> until now.`,
    ),
  },
  {
    id: "03-product",
    text: "Oakhaven City were losing for over forty minutes — their chances under ten percent. Then two late goals flipped everything. Sliding Doors doesn't just show you that moment — it replays it. Cancel that eighty-fourth-minute equalizer, and the engine re-simulates the rest of the match, hundreds of times. Dunmore's chances jump from nine percent... to ninety-two.",
    ssml: ssmlWrap(
      `Oakhaven City were losing for over forty minutes <break time="200ms"/> their chances under ten percent. Then two late goals flipped everything. <break time="400ms"/> Sliding Doors doesn't just show you that moment <break time="200ms"/> it replays it. Cancel that eighty-fourth-minute equalizer, and the engine re-simulates the rest of the match, hundreds of times. <break time="500ms"/> Dunmore's chances jump from nine percent <break time="600ms"/> <prosody pitch="+8%">to ninety-two.</prosody>`,
    ),
  },
  {
    id: "04-ask-ai",
    text: "Fans can just ask. An agent built on Microsoft Foundry turns the question into a real simulation — in English, or Hebrew.",
    ssml: ssmlWrap(
      `Fans can just ask. An agent built on Microsoft Foundry turns the question into a real simulation <break time="200ms"/> in English, or Hebrew.`,
    ),
  },
  {
    id: "05-under-the-hood",
    text: "Under the hood, two agents work together on Microsoft Foundry — an Analyst that calls the simulation, and a Narrator that explains it. Every number's grounded — nothing's invented.",
    ssml: ssmlWrap(
      `Under the hood, two agents work together on Microsoft Foundry <break time="200ms"/> an Analyst that calls the simulation, and a Narrator that explains it. <break time="250ms"/> Every number's grounded <break time="150ms"/> nothing's invented.`,
    ),
  },
  {
    id: "06-outro",
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
      // SDK quirk: a word immediately followed by an SSML <break> sometimes
      // reports .text as that word PLUS the rest of the sentence appended.
      // The timing is still correct for just that one word, so sanitize to
      // the first token -- the words that follow still get their own
      // (correct) events afterward, so this doesn't lose or duplicate data.
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
const toRun = requested.length ? scenes.filter((s) => requested.includes(s.id)) : scenes;
if (requested.length && toRun.length !== requested.length) {
  const found = toRun.map((s) => s.id);
  throw new Error(`Unknown scene id(s): ${requested.filter((r) => !found.includes(r))}`);
}

let totalChars = 0;
for (const scene of toRun) {
  const mp3Path = `${OUT_DIR}${scene.id}.mp3`;
  console.log(`Synthesizing ${scene.id} (${scene.text.length} chars)...`);
  const words = await synthesize(scene.ssml, mp3Path);

  const captions = words.map((w, i) => ({
    text: i === 0 ? w.text : ` ${w.text}`,
    startMs: Math.round(w.startMs),
    endMs: Math.round(w.endMs),
    timestampMs: Math.round((w.startMs + w.endMs) / 2),
    confidence: null,
  }));
  writeFileSync(`${OUT_DIR}${scene.id}.captions.json`, JSON.stringify(captions, null, 2));
  console.log(`  -> ${words.length} words captured, last ends at ${Math.round(words.at(-1).endMs)}ms`);
  totalChars += scene.text.length;
}
console.log(`Done. Total characters synthesized this run: ${totalChars} (~$${((totalChars / 1_000_000) * 16).toFixed(3)})`);
