// Records real footage of the live Sliding Doors app with Playwright.
// Produces two clips under public/recordings/:
//   desktop.webm -- the full walkthrough at 1920x1080
//   mobile.webm  -- a short mobile-viewport pass
// Also writes public/recordings/timestamps.json with ms-since-recording-start
// markers for each beat, used to trim the desktop clip into scenes in Remotion.
import { chromium } from "playwright";
import { mkdirSync, writeFileSync, rmSync, readdirSync, renameSync } from "fs";
import "dotenv/config";

const APP_URL =
  process.env.APP_URL ||
  "https://sliding-doors.happytree-e6502234.switzerlandnorth.azurecontainerapps.io/";

const OUT_DIR = new URL("../public/recordings/", import.meta.url).pathname;
rmSync(OUT_DIR, { recursive: true, force: true });
mkdirSync(OUT_DIR, { recursive: true });

const marks = {};
const t0 = Date.now();
const mark = (name) => {
  marks[name] = Date.now() - t0;
  console.log(`[${marks[name]}ms] ${name}`);
};

// Smoothly scroll to an element. Streamlit's main content lives in its own
// scrollable container (not `window`), so a manual window.scrollTo loop is a
// no-op here -- use the element's own native smooth scrollIntoView instead,
// which finds whatever the real scrollable ancestor is.
async function smoothScrollTo(page, locator, { waitMs = 1400 } = {}) {
  const handle = await locator.elementHandle();
  if (!handle) throw new Error("element not found for smooth scroll");
  await page.evaluate(
    (el) => el.scrollIntoView({ behavior: "smooth", block: "center" }),
    handle,
  );
  await page.waitForTimeout(waitMs);
}

async function typeSlowly(locator, text, delayMs = 45) {
  await locator.click();
  await locator.fill("");
  await locator.pressSequentially(text, { delay: delayMs });
}

const browser = await chromium.launch();

// ---------- Desktop walkthrough ----------
const desktopDir = `${OUT_DIR}_desktop_raw`;
const ctx = await browser.newContext({
  viewport: { width: 1920, height: 1080 },
  recordVideo: { dir: desktopDir, size: { width: 1920, height: 1080 } },
});
const page = await ctx.newPage();

mark("recording_start");
await page.goto(APP_URL, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(1500);
mark("page_loaded");
// The scoreboard sits below the three intro cards -- scroll to it for the hook.
await smoothScrollTo(page, page.getByText("Oakhaven City", { exact: true }).first(), {
  waitMs: 1200,
});
mark("scoreboard_visible");
await page.waitForTimeout(3500); // hold on the scoreboard for the hook

// Scroll to win-probability chart + turning points
await smoothScrollTo(page, page.getByText("Top 3 turning points", { exact: true }), {
  waitMs: 1800,
});
mark("win_prob_and_turning_points_visible");
await page.waitForTimeout(15000); // time to read the chart + 3 cards (sized to the voiceover line)

// Scroll to What if section
await smoothScrollTo(page, page.getByText("What if...?", { exact: true }), {
  waitMs: 1400,
});
mark("whatif_section_visible");
await page.waitForTimeout(800);

// Open Event dropdown (defaults to the 1' goal), pick the 84' goal.
// The selected value is rendered inside an <input>, so getByText can't see
// it -- the combobox inputs are stable in DOM order: 0=match picker,
// 1=Event, 2=Change, 3-4=Match recap tone/language, 5=chat text input,
// 6-7=Ask tone/language.
const comboboxes = page.locator('input[placeholder="Choose an option"]');
await comboboxes.nth(1).click();
await page.waitForTimeout(700);
mark("event_dropdown_open");
await page
  .getByText("84' — Goal — Milan Marek (Oakhaven City)", { exact: true })
  .click();
await page.waitForTimeout(900);
mark("event_84_selected");

await page.getByText("Run counterfactual", { exact: true }).click();
mark("run_counterfactual_clicked");
await page.waitForTimeout(7000);
mark("counterfactual_result_visible");
await page.waitForTimeout(5000); // hold on the 9% -> 92% result

// Scroll to agent chat
await smoothScrollTo(page, page.getByText("Ask a what-if question", { exact: true }), {
  waitMs: 1400,
});
mark("agent_section_visible");
await page.waitForTimeout(600);

const chatInput = page.locator(
  'input[placeholder="e.g. What if the 42nd-minute goal hadn\'t happened?"]',
);
// The language dropdown for the "Ask a what-if question" section is the
// last "Choose an option" combobox on the page (tone is second-to-last).
const askLanguageDropdown = page.locator('input[placeholder="Choose an option"]').last();

async function setAskLanguage(lang) {
  await askLanguageDropdown.click();
  await page.waitForTimeout(400);
  await page.getByText(lang, { exact: true }).last().click();
  await page.waitForTimeout(400);
}

// Make sure language = english for the EN question
await setAskLanguage("english");

mark("en_question_typing_start");
await typeSlowly(chatInput, "What if Dunmore had scored their big chance in the 26th minute?");
mark("en_question_typing_end");
await page.waitForTimeout(500);
await page.getByText("Ask", { exact: true }).click();
mark("en_ask_clicked");
await page.waitForTimeout(16000);
mark("en_answer_visible");
await page.waitForTimeout(4500);

// Switch to Hebrew
await setAskLanguage("hebrew");
mark("he_question_typing_start");
await typeSlowly(chatInput, "מה היה קורה אם השער של אוקהייבן בדקה ה-90 היה נפסל?");
mark("he_question_typing_end");
await page.waitForTimeout(500);
await page.getByText("Ask", { exact: true }).click();
mark("he_ask_clicked");
await page.waitForTimeout(18000);
mark("he_answer_visible");
await page.waitForTimeout(4500);

// Refusal case, back to English
await setAskLanguage("english");
mark("refusal_question_typing_start");
await typeSlowly(chatInput, "What if they'd made a substitution at halftime?");
mark("refusal_question_typing_end");
await page.waitForTimeout(500);
await page.getByText("Ask", { exact: true }).click();
mark("refusal_ask_clicked");
await page.waitForTimeout(16000);
mark("refusal_answer_visible");
await page.waitForTimeout(4500);

// Scroll back up to the club-trait cards for the "club profile" beat
await smoothScrollTo(page, page.getByText("Oakhaven City", { exact: true }).first(), {
  waitMs: 1500,
});
mark("club_profile_visible");
await page.waitForTimeout(7000);

mark("recording_end");
await ctx.close();

// ---------- Mobile pass ----------
const mobileDir = `${OUT_DIR}_mobile_raw`;
const ctxMobile = await browser.newContext({
  viewport: { width: 414, height: 896 },
  recordVideo: { dir: mobileDir, size: { width: 414, height: 896 } },
});
const pageMobile = await ctxMobile.newPage();
mark("mobile_recording_start");
await pageMobile.goto(APP_URL, { waitUntil: "networkidle", timeout: 60000 });
await pageMobile.waitForTimeout(1500);
mark("mobile_hero_visible");
for (let i = 0; i < 40; i++) {
  await pageMobile.evaluate((y) => {
    document.querySelector('[data-testid="stMain"]').scrollBy(0, y);
  }, 60);
  await pageMobile.waitForTimeout(90);
}
mark("mobile_scroll_done");
await pageMobile.waitForTimeout(1500);
mark("mobile_recording_end");
await ctxMobile.close();

await browser.close();

// Playwright names video files by an internal id; move them to stable names.
function moveLatestVideo(dir, destName) {
  const files = readdirSync(dir).filter((f) => f.endsWith(".webm"));
  if (files.length !== 1) {
    throw new Error(`expected exactly 1 video in ${dir}, found ${files.length}`);
  }
  renameSync(`${dir}/${files[0]}`, `${OUT_DIR}${destName}`);
  rmSync(dir, { recursive: true, force: true });
}
moveLatestVideo(desktopDir, "desktop.webm");
moveLatestVideo(mobileDir, "mobile.webm");

writeFileSync(`${OUT_DIR}timestamps.json`, JSON.stringify(marks, null, 2));
console.log("Done. Timestamps written to public/recordings/timestamps.json");
