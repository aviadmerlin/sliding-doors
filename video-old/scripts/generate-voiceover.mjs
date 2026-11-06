// Generates the English neural-voice narration for each scene with Azure AI
// Speech, writing one MP3 per scene to public/audio/. Remotion's
// calculateMetadata measures these files to size each scene.
import sdk from "microsoft-cognitiveservices-speech-sdk";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import "dotenv/config";

const KEY = process.env.AZURE_SPEECH_KEY;
const REGION = process.env.AZURE_SPEECH_REGION;
if (!KEY || !REGION) {
  throw new Error("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in video/.env");
}

const VOICE = "en-US-AndrewNeural";
const OUT_DIR = new URL("../public/audio/", import.meta.url).pathname;
rmSync(OUT_DIR, { recursive: true, force: true });
mkdirSync(OUT_DIR, { recursive: true });

// Numbers (9% -> 92%, 91% -> 9%, etc.) are read off the real run captured in
// public/recordings/desktop.webm -- see public/recordings/timestamps.json.
const scenes = [
  {
    id: "01-hook",
    text: `Every fan has had this argument: "If that goal hadn't gone in, we'd have won." Until now, it was just opinion. Sliding Doors answers it with data.`,
  },
  {
    id: "02-turning-points",
    text: `Oakhaven City were losing 2 to 1 for over forty minutes. Their chances dropped below 10%. Then an equalizer at 84 and a winner at 90 flipped everything. Sliding Doors ranks the turning points by how much each moment really changed the outcome.`,
  },
  {
    id: "03-whatif",
    text: `Now replay it. What if the 84th-minute equalizer had been cancelled? The engine re-simulates the rest of the match hundreds of times, and Dunmore's chances jump from 9% to 92%.`,
  },
  {
    id: "04-agent",
    text: `Fans can just ask, in English or Hebrew. An agent built with Microsoft Agent Framework, running gpt-5-mini on Microsoft Foundry, turns the question into a call to the simulation engine and explains the result. Every number comes from the engine, never invented. And if a question can't be simulated, it says so honestly.`,
  },
  {
    id: "05-calibration",
    text: `Under the hood: five seasons of synthetic history, 1,900 matches. Each club's character is learned from the data, and the win probabilities are validated on matches the model never saw.`,
  },
  {
    id: "06-outro",
    text: `Sliding Doors. Every match, every moment, every what if.`,
  },
];

function synthesize(text, outPath) {
  return new Promise((resolve, reject) => {
    const speechConfig = sdk.SpeechConfig.fromSubscription(KEY, REGION);
    speechConfig.speechSynthesisVoiceName = VOICE;
    speechConfig.speechSynthesisOutputFormat =
      sdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3;
    const audioConfig = sdk.AudioConfig.fromAudioFileOutput(outPath);
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig);
    synthesizer.speakTextAsync(
      text,
      (result) => {
        synthesizer.close();
        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          resolve();
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

let totalChars = 0;
for (const scene of scenes) {
  const outPath = `${OUT_DIR}${scene.id}.mp3`;
  console.log(`Synthesizing ${scene.id} (${scene.text.length} chars)...`);
  await synthesize(scene.text, outPath);
  totalChars += scene.text.length;
}
console.log(`Done. Total characters: ${totalChars} (~$${((totalChars / 1_000_000) * 16).toFixed(3)} at $16/1M chars)`);
