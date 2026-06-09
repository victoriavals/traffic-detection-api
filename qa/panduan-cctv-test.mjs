/**
 * Panduan CCTV regression test — Playwright.
 *
 * Verifies the new /panduan-cctv guide:
 *  1. Public access — renders without login on both desktop + mobile.
 *  2. All sections render (hero, examples, syarat, tips, hindari, FAQ, CTA).
 *  3. Sidebar (when logged in) shows the new "Panduan & Bantuan" section
 *     header, the new "Panduan CCTV" item with `nav-panduan-cctv` tour anchor,
 *     and the renamed "Tour Interaktif" label.
 *  4. Entry points work:
 *       - Landing page CCTV nudge card → /panduan-cctv
 *       - /panduan TourLibrary shortcut card → /panduan-cctv
 *       - /live-monitoring inline link → /panduan-cctv
 *  5. "Coba deteksi" button opens the demo modal (loading state visible).
 *  6. No horizontal overflow on mobile portrait + landscape.
 *
 * Run:
 *   node d:\computer-vision\traffic-detection-api\qa\panduan-cctv-test.mjs
 */

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SCREENSHOTS_DIR = join(__dirname, "screenshots");
const FRONTEND = process.env.FRONTEND_URL || "http://localhost:9321";

const DESKTOP = { width: 1280, height: 800 };
const MOBILE = { width: 375, height: 667 };
const LANDSCAPE = { width: 667, height: 375 };

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

async function seedTours(page) {
  await page.evaluate(() => {
    const now = new Date().toISOString();
    localStorage.setItem("traffic_onboarding_done_v2_operator", now);
    localStorage.setItem("traffic_onboarding_done_v2_admin", now);
    for (const id of ["lokasi", "proses-video", "laporan"])
      localStorage.setItem(`traffic_tour_page_v1_${id}`, now);
  });
}

async function registerAndLogin(page) {
  const rnd = Math.random().toString(36).slice(2, 8);
  await page.goto(`${FRONTEND}/register`, { waitUntil: "domcontentloaded" });
  await seedTours(page);
  await page.fill('input[id="name"]', `CCTV Audit ${rnd}`);
  await page.fill('input[id="email"]', `cctvaudit_${rnd}@test.local`);
  await page.fill('input[id="password"]', "TestPass123!");
  await page.fill('input[id="confirm"]', "TestPass123!");
  await page.click('button[type="submit"]');
  await page.waitForURL(/\/dashboard$/, { timeout: 10000 });
  await page.waitForTimeout(800);
}

async function main() {
  await mkdir(SCREENSHOTS_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });

  try {
    /* ── 1. PUBLIC ACCESS (no login) — desktop ─────────────────────── */
    {
      const ctx = await browser.newContext({ viewport: DESKTOP });
      const page = await ctx.newPage();

      await page.goto(`${FRONTEND}/panduan-cctv`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(800);

      record(
        "PUBLIC-001",
        "/panduan-cctv loads without login (desktop)",
        page.url().endsWith("/panduan-cctv"),
        `final url=${page.url()}`,
      );

      // Hero must contain the page title.
      const h1 = (await page.locator("h1").first().textContent()) || "";
      record(
        "PUBLIC-002",
        "Page hero shows 'Panduan Posisi CCTV'",
        h1.toLowerCase().includes("panduan posisi cctv"),
        `h1="${h1.trim()}"`,
      );

      // All major sections render (look for distinctive heading text).
      const headings = await page.locator("h2").allTextContents();
      const headingTxt = headings.join(" | ").toLowerCase();
      const sectionsOk =
        headingTxt.includes("contoh pov yang baik") &&
        headingTxt.includes("syarat cctv") &&
        headingTxt.includes("tips pemasangan") &&
        headingTxt.includes("contoh pov yang buruk") &&
        headingTxt.includes("daftar lengkap") &&
        headingTxt.includes("pertanyaan umum");
      record(
        "PUBLIC-003",
        "All major sections render (Baik / Syarat / Tips / Buruk / Daftar / FAQ)",
        sectionsOk,
        `h2s=${headings.length}`,
      );

      // "Coba deteksi" button only on the 2 good example cards (not bad).
      const tryButtons = await page
        .locator("button:has-text('Coba deteksi')")
        .count();
      record(
        "PUBLIC-004",
        "Only 2 'Coba deteksi' buttons (good examples only)",
        tryButtons === 2,
        `count=${tryButtons}`,
      );

      // 4 example article cards render (2 good + 2 bad). Each card contains
      // exactly one lazy-loaded image, so counting those identifies the cards
      // unambiguously even when "IDEAL"/"HINDARI" appear as substring in copy.
      const cardImages = await page.locator('article img[loading="lazy"]').count();
      record(
        "PUBLIC-004b",
        "4 example cards render (2 good + 2 bad)",
        cardImages === 4,
        `cards=${cardImages}`,
      );

      // CTA section shows public-mode buttons (Daftar & Mulai / Login).
      const daftarBtn = await page
        .locator("a:has-text('Daftar & Mulai')")
        .count();
      record(
        "PUBLIC-005",
        "CTA shows public-mode buttons (Daftar & Mulai)",
        daftarBtn >= 1,
      );

      await page.screenshot({
        path: join(SCREENSHOTS_DIR, "panduan-cctv-desktop.png"),
        fullPage: true,
      });

      /* ── 2. Demo modal opens on "Coba deteksi" click ───────────────── */
      // Click the first try-detection button. Modal opens with loading state
      // even if the backend isn't reachable.
      await page.locator("button:has-text('Coba deteksi')").first().click();
      await page.waitForTimeout(500);
      const dialogVisible = await page
        .locator('[role="dialog"]')
        .first()
        .isVisible()
        .catch(() => false);
      record("PUBLIC-006", "'Coba deteksi' opens the demo dialog", dialogVisible);

      await page.screenshot({
        path: join(SCREENSHOTS_DIR, "panduan-cctv-demo-modal.png"),
      });

      // Close dialog.
      await page.keyboard.press("Escape");
      await page.waitForTimeout(300);

      await ctx.close();
    }

    /* ── 3. RESPONSIVENESS — mobile portrait + landscape ──────────────── */
    {
      const ctx = await browser.newContext({ viewport: MOBILE });
      const page = await ctx.newPage();
      await page.goto(`${FRONTEND}/panduan-cctv`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(700);

      record(
        "RESP-001",
        "/panduan-cctv: no horizontal overflow on mobile portrait",
        await checkNoHorizontalOverflow(page),
      );

      await page.screenshot({
        path: join(SCREENSHOTS_DIR, "panduan-cctv-mobile.png"),
        fullPage: true,
      });
      await ctx.close();
    }

    {
      const ctx = await browser.newContext({ viewport: LANDSCAPE });
      const page = await ctx.newPage();
      await page.goto(`${FRONTEND}/panduan-cctv`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(700);

      record(
        "RESP-002",
        "/panduan-cctv: no horizontal overflow on mobile landscape",
        await checkNoHorizontalOverflow(page),
      );

      // Body must be vertically scrollable (content overflows 375h).
      const scrollable = await page.evaluate(
        () => document.documentElement.scrollHeight > window.innerHeight + 1,
      );
      record(
        "RESP-003",
        "/panduan-cctv: content is vertically scrollable in landscape",
        scrollable,
      );
      await ctx.close();
    }

    /* ── 4. ENTRY POINTS ───────────────────────────────────────────────── */
    // Landing page nudge card → /panduan-cctv
    {
      const ctx = await browser.newContext({ viewport: DESKTOP });
      const page = await ctx.newPage();
      await page.goto(`${FRONTEND}/`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(600);
      const nudge = page.locator("a:has-text('Lihat Panduan')").first();
      const nudgeVisible = await nudge.isVisible().catch(() => false);
      record("ENTRY-001", "Landing page shows 'Lihat Panduan' CTA card", nudgeVisible);
      if (nudgeVisible) {
        await nudge.click();
        await page.waitForURL(/\/panduan-cctv$/, { timeout: 5000 });
        record(
          "ENTRY-002",
          "Landing CTA navigates to /panduan-cctv",
          page.url().endsWith("/panduan-cctv"),
        );
      } else {
        record("ENTRY-002", "Landing CTA navigates to /panduan-cctv", false, "nudge not visible");
      }
      await ctx.close();
    }

    /* ── 5. SIDEBAR + AUTHENTICATED ENTRY POINTS ───────────────────────── */
    {
      const ctx = await browser.newContext({ viewport: DESKTOP });
      const page = await ctx.newPage();
      await registerAndLogin(page);

      // Sidebar has the new "Panduan & Bantuan" group header.
      const sidebarHeader = await page
        .locator("aside[data-tour='sidebar']")
        .locator("text=Panduan & Bantuan")
        .count();
      record(
        "SIDEBAR-001",
        "Sidebar shows 'Panduan & Bantuan' section header",
        sidebarHeader >= 1,
      );

      // New nav item exists.
      const cctvNav = await page
        .locator("[data-tour='nav-panduan-cctv']")
        .count();
      record(
        "SIDEBAR-002",
        "Sidebar has 'nav-panduan-cctv' item (Panduan CCTV)",
        cctvNav >= 1,
      );

      // Tour Interaktif rename — old "Panduan" label replaced.
      const tourInteraktif = await page
        .locator("aside[data-tour='sidebar']")
        .locator("text=Tour Interaktif")
        .count();
      record(
        "SIDEBAR-003",
        "Sidebar shows renamed 'Tour Interaktif' label",
        tourInteraktif >= 1,
      );

      // Click nav-panduan-cctv → navigate to /panduan-cctv
      await page.locator("[data-tour='nav-panduan-cctv']").click();
      await page.waitForURL(/\/panduan-cctv$/, { timeout: 5000 });
      record(
        "SIDEBAR-004",
        "Clicking sidebar item navigates to /panduan-cctv",
        page.url().endsWith("/panduan-cctv"),
      );

      // CTA shows logged-in mode (Buat Lokasi instead of Daftar)
      const buatLokasi = await page
        .locator("a:has-text('Buat Lokasi')")
        .count();
      record(
        "SIDEBAR-005",
        "CTA section shows logged-in actions (Buat Lokasi)",
        buatLokasi >= 1,
      );

      // /panduan TourLibrary shortcut card → /panduan-cctv
      await page.goto(`${FRONTEND}/panduan`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(800);
      const shortcutCard = page.locator("a[href='/panduan-cctv']").first();
      const shortcutVisible = await shortcutCard.isVisible().catch(() => false);
      record(
        "ENTRY-003",
        "/panduan shows shortcut card to Panduan CCTV",
        shortcutVisible,
      );

      // /live-monitoring inline link → /panduan-cctv
      await page.goto(`${FRONTEND}/live-monitoring`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(700);
      const liveLinkCount = await page
        .locator("a[href='/panduan-cctv']")
        .count();
      record(
        "ENTRY-004",
        "/live-monitoring shows inline 'Cek panduan' link",
        liveLinkCount >= 1,
      );

      await page.screenshot({
        path: join(SCREENSHOTS_DIR, "live-monitoring-cctv-link.png"),
        fullPage: false,
      });

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
