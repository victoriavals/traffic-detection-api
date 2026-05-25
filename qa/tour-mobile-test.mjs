/**
 * Tour-on-mobile regression test — Playwright.
 *
 * Reproduces the reported bug: starting the *overview* tour on a phone left
 * the sidebar drawer closed, so its nav-item steps ("Mulai dari Lokasi", etc.)
 * pointed at elements that were translated off-screen. The popover floated over
 * an empty page.
 *
 * Verifies the fix:
 *  1. Starting the overview tour force-opens the mobile drawer.
 *  2. The highlighted nav item (`nav-lokasi`) lands fully inside the viewport.
 *  3. Finishing/closing the tour collapses the drawer again.
 *  4. /panduan itself has no horizontal overflow on mobile (responsiveness).
 *
 * Run (node_modules junction already points at the frontend install):
 *   node d:\computer-vision\traffic-detection-api\qa\tour-mobile-test.mjs
 */

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SCREENSHOTS_DIR = join(__dirname, "screenshots");
const FRONTEND = process.env.FRONTEND_URL || "http://localhost:9321";

const MOBILE = { width: 375, height: 667 };

const results = [];
function record(id, title, ok, detail = "") {
  const tag = ok ? "✅ PASS" : "❌ FAIL";
  console.log(`[${tag}] ${id} — ${title}${detail ? " :: " + detail : ""}`);
  results.push({ id, title, ok, detail });
}

async function checkNoHorizontalOverflow(page) {
  return !(await page.evaluate(
    () => document.body.scrollWidth > document.body.clientWidth + 1,
  ));
}

// Drawer is "open" when its computed transform has no negative X translation.
async function drawerOpen(page) {
  const t = await page
    .locator('aside[data-tour="sidebar"]')
    .evaluate((el) => getComputedStyle(el).transform)
    .catch(() => "none");
  // open == translateX(0) → "none" or matrix(...,0,0); closed == matrix(...,-280,0)
  return t === "none" || /matrix\(1, 0, 0, 1, 0,/.test(t);
}

async function registerAndLogin(page) {
  const rnd = Math.random().toString(36).slice(2, 8);
  const email = `tourtest_${rnd}@test.local`;
  await page.goto(`${FRONTEND}/register`, { waitUntil: "domcontentloaded" });
  // Dismiss the auto-onboarding so it doesn't race our manual trigger.
  await page.evaluate(() => {
    const now = new Date().toISOString();
    localStorage.setItem("traffic_onboarding_done_v2_operator", now);
    localStorage.setItem("traffic_onboarding_done_v2_admin", now);
    for (const id of ["lokasi", "proses-video", "laporan"])
      localStorage.setItem(`traffic_tour_page_v1_${id}`, now);
  });
  await page.fill('input[id="name"]', `Tour Test ${rnd}`);
  await page.fill('input[id="email"]', email);
  await page.fill('input[id="password"]', "TestPass123!");
  await page.fill('input[id="confirm"]', "TestPass123!");
  await page.click('button[type="submit"]');
  await page.waitForURL(/\/dashboard$/, { timeout: 10000 });
  await page.waitForTimeout(1200);
  return email;
}

async function main() {
  await mkdir(SCREENSHOTS_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: MOBILE });
  const page = await ctx.newPage();

  try {
    await registerAndLogin(page);

    // ── /panduan responsiveness ──────────────────────────────────────────
    await page.goto(`${FRONTEND}/panduan`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(900);
    record("PD-001", "Panduan has no horizontal overflow on mobile",
      await checkNoHorizontalOverflow(page));
    // Tour library cards collapse to a single column (grid still renders).
    const tourCards = await page.locator("section[aria-label='Tur Interaktif'] button").count();
    record("PD-002", "Panduan tour library renders cards", tourCards >= 1,
      `cards=${tourCards}`);
    await page.screenshot({ path: join(SCREENSHOTS_DIR, "mobile-panduan.png"), fullPage: true });

    // ── Start the overview tour from Panduan ─────────────────────────────
    await page.locator('button:has-text("Mulai Ulang Tour Overview")').click();
    await page.waitForSelector(".driver-popover", { timeout: 5000 });
    await page.waitForTimeout(500); // let the drawer slide in

    record("TR-001", "Overview tour starts (driver popover visible)",
      await page.locator(".driver-popover").isVisible());

    // Drawer should be force-opened immediately at tour start.
    record("TR-002", "Mobile drawer force-opens when tour starts",
      await drawerOpen(page));

    // Advance: step1 welcome → step2 sidebar → step3 nav-lokasi.
    const next = page.locator(".driver-popover-next-btn");
    await next.click();
    await page.waitForTimeout(400);
    await next.click();
    await page.waitForTimeout(500);

    const title = (await page.locator(".driver-popover-title").textContent()) || "";
    record("TR-003", "Reached the 'Mulai dari Lokasi' sidebar step",
      title.includes("Lokasi"), `title="${title.trim()}"`);

    // The highlighted nav item must be fully inside the viewport now.
    const box = await page.locator("[data-tour='nav-lokasi']").boundingBox();
    const inView =
      !!box &&
      box.x >= 0 &&
      box.y >= 0 &&
      box.x + box.width <= MOBILE.width + 1 &&
      box.y + box.height <= MOBILE.height + 1;
    record("TR-004", "nav-lokasi is on-screen during its tour step", inView,
      box ? `box=${Math.round(box.x)},${Math.round(box.y)} ${Math.round(box.width)}×${Math.round(box.height)}` : "no box");

    await page.screenshot({ path: join(SCREENSHOTS_DIR, "mobile-tour-nav-lokasi.png"), fullPage: false });

    // ── Close the tour → drawer collapses again ──────────────────────────
    await page.keyboard.press("Escape");
    await page.waitForTimeout(600);
    record("TR-005", "Drawer collapses after the tour ends",
      !(await drawerOpen(page)));
  } finally {
    await ctx.close();
    await browser.close();
  }

  const passed = results.filter((r) => r.ok).length;
  console.log(`\n${"━".repeat(50)}`);
  console.log(`📊 ${passed}/${results.length} passed`);
  console.log(`${"━".repeat(50)}\n`);
  process.exit(results.length - passed);
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
