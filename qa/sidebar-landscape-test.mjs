/**
 * Sidebar scrollability regression test — Playwright.
 *
 * Bug: on a phone in *landscape* orientation the viewport height (~375px)
 * is too short to fit all 9 nav items + admin/konfigurasi panels + footer.
 * The previous layout had `shrink-0` on <nav> and `overflow-hidden` on the
 * <aside>, so the bottom of the menu was clipped with no way to scroll
 * (Panduan, Bantuan, the user-profile/footer all unreachable).
 *
 * Fix: nav + admin/konfigurasi share one `flex-1 min-h-0 overflow-y-auto`
 * region so the whole menu scrolls as one. This test verifies it.
 *
 * Tests both orientations to confirm portrait was not regressed.
 *
 * Run:
 *   node d:\computer-vision\traffic-detection-api\qa\sidebar-landscape-test.mjs
 */

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SCREENSHOTS_DIR = join(__dirname, "screenshots");
const FRONTEND = process.env.FRONTEND_URL || "http://localhost:9321";

// iPhone SE-ish dimensions.
const PORTRAIT = { width: 375, height: 667 };
const LANDSCAPE = { width: 667, height: 375 };

const results = [];
function record(id, title, ok, detail = "") {
  const tag = ok ? "✅ PASS" : "❌ FAIL";
  console.log(`[${tag}] ${id} — ${title}${detail ? " :: " + detail : ""}`);
  results.push({ id, title, ok, detail });
}

async function registerAndLogin(page) {
  const rnd = Math.random().toString(36).slice(2, 8);
  await page.goto(`${FRONTEND}/register`, { waitUntil: "domcontentloaded" });
  await page.evaluate(() => {
    const now = new Date().toISOString();
    localStorage.setItem("traffic_onboarding_done_v2_operator", now);
    localStorage.setItem("traffic_onboarding_done_v2_admin", now);
    for (const id of ["lokasi", "proses-video", "laporan"])
      localStorage.setItem(`traffic_tour_page_v1_${id}`, now);
  });
  await page.fill('input[id="name"]', `Landscape ${rnd}`);
  await page.fill('input[id="email"]', `landscape_${rnd}@test.local`);
  await page.fill('input[id="password"]', "TestPass123!");
  await page.fill('input[id="confirm"]', "TestPass123!");
  await page.click('button[type="submit"]');
  await page.waitForURL(/\/dashboard$/, { timeout: 10000 });
  await page.waitForTimeout(800);
}

async function openDrawer(page) {
  await page.locator('button[aria-label="Buka menu navigasi"]').click();
  await page.waitForTimeout(400);
}

// Returns the scroll container inside the sidebar (the new flex-1 min-h-0
// wrapper), and the bounding box of the "Bantuan" nav link (last menu item).
async function probeSidebar(page) {
  return await page.evaluate(() => {
    const aside = document.querySelector('aside[data-tour="sidebar"]');
    if (!aside) return { ok: false, reason: "no aside" };
    // Find the scroll container — the one with overflow-y: auto and a vertical
    // scroll content larger than its visible area (or capable of scrolling).
    let scroller = null;
    for (const el of aside.querySelectorAll("div")) {
      const cs = getComputedStyle(el);
      if (cs.overflowY === "auto" || cs.overflowY === "scroll") {
        scroller = el;
        break;
      }
    }
    const last = aside.querySelector("a[href='/bantuan']");
    return {
      ok: true,
      hasScroller: !!scroller,
      scrollHeight: scroller?.scrollHeight ?? 0,
      clientHeight: scroller?.clientHeight ?? 0,
      lastVisible: last
        ? (() => {
            const r = last.getBoundingClientRect();
            const ar = aside.getBoundingClientRect();
            return {
              top: r.top,
              bottom: r.bottom,
              asideBottom: ar.bottom,
              asideTop: ar.top,
            };
          })()
        : null,
    };
  });
}

async function main() {
  await mkdir(SCREENSHOTS_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });

  try {
    // ── LANDSCAPE ────────────────────────────────────────────────────────
    {
      const ctx = await browser.newContext({ viewport: LANDSCAPE });
      const page = await ctx.newPage();
      await registerAndLogin(page);
      await openDrawer(page);

      const probe = await probeSidebar(page);
      record("LS-001", "Sidebar has a scroll container in landscape",
        probe.ok && probe.hasScroller,
        `scrollH=${probe.scrollHeight} clientH=${probe.clientHeight}`);

      // The whole point of the fix: the scroller's content must exceed its
      // visible area in landscape (otherwise the menu is already short
      // enough; this would mean the test viewport changed).
      record("LS-002", "Scroll container actually overflows in landscape",
        probe.scrollHeight > probe.clientHeight,
        `${probe.scrollHeight} > ${probe.clientHeight}`);

      // The bottom nav item ("Bantuan") must be scrollable into view.
      const bantuan = page.locator("aside a[href='/bantuan']");
      await bantuan.scrollIntoViewIfNeeded({ timeout: 2000 });
      await page.waitForTimeout(200);
      const visible = await bantuan.isVisible();
      const box = await bantuan.boundingBox();
      const bottom = box ? box.y + box.height : null;
      const insideAside = box ? bottom <= LANDSCAPE.height + 1 && box.y >= 0 : false;
      record("LS-003", "Bantuan link scrolls into view in landscape",
        visible && insideAside,
        box ? `top=${Math.round(box.y)} bottom=${Math.round(bottom)}` : "no box");

      await page.screenshot({
        path: join(SCREENSHOTS_DIR, "mobile-landscape-sidebar.png"),
        fullPage: false,
      });

      await ctx.close();
    }

    // ── PORTRAIT (regression: must still work) ──────────────────────────
    {
      const ctx = await browser.newContext({ viewport: PORTRAIT });
      const page = await ctx.newPage();
      await registerAndLogin(page);
      await openDrawer(page);

      const bantuan = page.locator("aside a[href='/bantuan']");
      await bantuan.scrollIntoViewIfNeeded({ timeout: 2000 }).catch(() => {});
      await page.waitForTimeout(200);
      const visible = await bantuan.isVisible();
      const box = await bantuan.boundingBox();
      const bottom = box ? box.y + box.height : null;
      const insideAside = box ? bottom <= PORTRAIT.height + 1 && box.y >= 0 : false;
      record("PT-001", "Bantuan link reachable in portrait (regression)",
        visible && insideAside,
        box ? `top=${Math.round(box.y)} bottom=${Math.round(bottom)}` : "no box");

      // Footer (user profile + version) still pinned at the bottom in portrait.
      const footerBox = await page.locator("aside [data-tour='user-profile']").boundingBox().catch(() => null);
      const footerBottom = footerBox ? footerBox.y + footerBox.height : null;
      record("PT-002", "User-profile footer is pinned at bottom in portrait",
        !!footerBox && footerBottom >= PORTRAIT.height - 80,
        footerBox ? `bottom=${Math.round(footerBottom)}` : "no box");

      await ctx.close();
    }
  } finally {
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
