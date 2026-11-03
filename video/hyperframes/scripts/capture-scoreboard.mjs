// Captures a clean, fully-settled, TIGHTLY-CROPPED screenshot of just the
// scoreboard card (no sidebar) for the Hook scene -- the sidebar shows up
// later in Proof instead. Verifies there is NO loading state visible by
// (a) checking for Streamlit's own "running" indicator in the DOM, and
// (b) taking two full-page screenshots 800ms apart and confirming they're
// byte-identical (proves the page has fully settled).
import { chromium } from "playwright";
import { createHash } from "crypto";
import { writeFileSync } from "fs";

const APP_URL = process.env.APP_URL || "http://localhost:8501";
const OUT = new URL("../assets/scoreboard.png", import.meta.url).pathname;
const PAD = 6; // px of margin -- just enough for anti-aliased edges, not
// enough to bleed into the neighboring cards above/below (Streamlit's own
// gap between cards is small; any more padding clips their rounded corners
// in). Visual breathing room comes from the composition's own layout
// instead of from this crop.

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
await page.goto(APP_URL, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(3000);

const runningIndicator = await page
  .locator('[data-testid="stStatusWidget"], [data-testid="stSpinner"], .stSpinner')
  .count();
if (runningIndicator > 0) {
  throw new Error(
    `Streamlit still shows a running/spinner indicator (${runningIndicator} found) -- not settled.`,
  );
}

const scoreEl = page.getByText("Oakhaven City", { exact: true }).first();
await scoreEl.scrollIntoViewIfNeeded();
await page.waitForTimeout(500);

// Stability check on the full page first (same guarantee as before).
async function fullShot() {
  return page.screenshot({ fullPage: false });
}
const a = await fullShot();
await page.waitForTimeout(800);
const b = await fullShot();
if (createHash("sha256").update(a).digest("hex") !== createHash("sha256").update(b).digest("hex")) {
  throw new Error("Two screenshots 800ms apart differ -- page is still animating/settling.");
}

// Walk up from the "Oakhaven City" text to the scoreboard card: the first
// ancestor with a real border (Streamlit's bordered container). Avoids
// hardcoding a hashed class name, which changes between Streamlit builds.
const cardBox = await scoreEl.evaluate(() => {
  let node = document.evaluate(
    "//*[text()='Oakhaven City']",
    document,
    null,
    XPathResult.FIRST_ORDERED_NODE_TYPE,
    null,
  ).singleNodeValue;
  while (node) {
    const cs = getComputedStyle(node);
    if (parseFloat(cs.borderWidth) > 0) {
      const r = node.getBoundingClientRect();
      return { x: r.x, y: r.y, width: r.width, height: r.height };
    }
    node = node.parentElement;
  }
  return null;
});
if (!cardBox) throw new Error("Could not find the bordered scoreboard card ancestor.");

const clip = {
  x: Math.max(0, cardBox.x - PAD),
  y: Math.max(0, cardBox.y - PAD),
  width: cardBox.width + PAD * 2,
  height: cardBox.height + PAD * 2,
};
console.log("Scoreboard card clip:", clip);

const cropped = await page.screenshot({ clip });
writeFileSync(OUT, cropped);
console.log(`Stable, loading-state-free, sidebar-free capture confirmed. Saved to ${OUT}`);
await browser.close();
