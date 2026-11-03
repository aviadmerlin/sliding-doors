// Records the real Proof-beat interaction against the local app instance:
// scoreboard (sidebar visible this time), the What-if panel (pick the 84'
// goal, run the counterfactual, real 9%->92% result), then the Ask box
// with the real EN and HE questions/answers. Produces one continuous
// video + a timestamps.json (ms since recording start) for cutting scene
// beats in assets/proof/.
import { chromium } from "playwright";
import { mkdirSync, writeFileSync, rmSync, readdirSync, renameSync } from "fs";

const APP_URL = process.env.APP_URL || "http://localhost:8501";
const OUT_DIR = new URL("../assets/proof/", import.meta.url).pathname;
rmSync(OUT_DIR, { recursive: true, force: true });
mkdirSync(OUT_DIR, { recursive: true });

const marks = {};
const t0 = Date.now();
const mark = (name) => {
  marks[name] = Date.now() - t0;
  console.log(`[${marks[name]}ms] ${name}`);
};

async function typeSlowly(locator, text, delayMs = 42) {
  await locator.click();
  await locator.fill("");
  await locator.pressSequentially(text, { delay: delayMs });
}

const browser = await chromium.launch();
const rawDir = `${OUT_DIR}_raw`;
const ctx = await browser.newContext({
  viewport: { width: 1920, height: 1080 },
  recordVideo: { dir: rawDir, size: { width: 1920, height: 1080 } },
});
const page = await ctx.newPage();

mark("recording_start");
await page.goto(APP_URL, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(2000);
mark("scoreboard_visible");
await page.waitForTimeout(1500);

// Scroll to What-if panel.
await page.getByText("What if...?", { exact: true }).scrollIntoViewIfNeeded();
await page.waitForTimeout(500);
mark("whatif_visible");

const comboboxes = page.locator('input[placeholder="Choose an option"]');
await comboboxes.nth(1).click();
await page.waitForTimeout(600);
mark("event_dropdown_open");
await page.getByText("84' — Goal — Milan Marek (Oakhaven City)", { exact: true }).click();
await page.waitForTimeout(700);
mark("event_84_selected");

await page.getByText("Run counterfactual", { exact: true }).click();
mark("run_clicked");
await page.waitForTimeout(7000);
mark("result_visible");
await page.waitForTimeout(3000);

// Scroll to the Ask box.
await page.getByText("Ask a what-if question", { exact: true }).scrollIntoViewIfNeeded();
await page.waitForTimeout(500);
mark("ask_box_visible");

const chatInput = page.locator('input[placeholder="e.g. What if the 42nd-minute goal hadn\'t happened?"]');
const langDropdown = page.locator('input[placeholder="Choose an option"]').last();
async function setLang(lang) {
  await langDropdown.click();
  await page.waitForTimeout(350);
  await page.getByText(lang, { exact: true }).last().click();
  await page.waitForTimeout(350);
}

await setLang("english");
mark("en_typing_start");
await typeSlowly(chatInput, "What if Dunmore had scored their big chance in the 26th minute?");
mark("en_typing_end");
await page.waitForTimeout(400);
await page.getByText("Ask", { exact: true }).click();
mark("en_ask_clicked");
await page.waitForTimeout(16000);
mark("en_answer_visible");
await page.waitForTimeout(3000);

await setLang("hebrew");
mark("he_typing_start");
await typeSlowly(chatInput, "מה היה קורה אם השער של אוקהייבן בדקה ה-90 היה נפסל?");
mark("he_typing_end");
await page.waitForTimeout(400);
await page.getByText("Ask", { exact: true }).click();
mark("he_ask_clicked");
await page.waitForTimeout(18000);
mark("he_answer_visible");
await page.waitForTimeout(3000);

mark("recording_end");
await ctx.close();
await browser.close();

function moveLatestVideo(dir, destName) {
  const files = readdirSync(dir).filter((f) => f.endsWith(".webm"));
  if (files.length !== 1) throw new Error(`expected 1 video in ${dir}, found ${files.length}`);
  renameSync(`${dir}/${files[0]}`, `${OUT_DIR}${destName}`);
  rmSync(dir, { recursive: true, force: true });
}
moveLatestVideo(rawDir, "proof.webm");

const t0mark = marks.recording_start;
const rel = {};
for (const [k, v] of Object.entries(marks)) rel[k] = Math.round(((v - t0mark) / 1000) * 1000) / 1000;
writeFileSync(`${OUT_DIR}marks.json`, JSON.stringify(rel, null, 2));
console.log("Done. marks.json written.");
