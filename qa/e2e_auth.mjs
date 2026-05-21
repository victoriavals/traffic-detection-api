/**
 * Real-browser E2E test for Register + Login.
 * Drives a headless Chromium against http://localhost:9321.
 *
 * Usage (see qa/README.md "Frontend E2E" section for NODE_PATH/symlink setup):
 *   node traffic-detection-api/qa/e2e_auth.mjs
 *
 * What this proves (that unit tests + curl cannot):
 *   - Real browser CORS behaviour (preflight OPTIONS, cookie/credential rules)
 *   - Vite dev proxy works end-to-end for browser requests
 *   - Register form actually persists user, redirects to /dashboard
 *   - Login form authenticates the same user
 *   - Network errors surface as friendly Indonesian messages, not "Failed to fetch"
 */

import { chromium } from "playwright";

const FRONTEND = process.env.FRONTEND_URL || "http://localhost:9321";
const RND = Math.random().toString(36).slice(2, 8);
const EMAIL = `e2e_${RND}@test.local`;
const NAME = `E2E ${RND}`;
const PASSWORD = "Pass12345";

const results = [];
function record(name, ok, detail = "") {
  const tag = ok ? "PASS" : "FAIL";
  console.log(`[${tag}] ${name}${detail ? " -- " + detail : ""}`);
  results.push({ name, ok, detail });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function cleanupUser(email) {
  // Best-effort: delete via direct backend call so subsequent runs are clean.
  try {
    const { MongoClient } = await import("mongodb");
    const uri = process.env.MONGO_URI;
    const dbName = process.env.MONGO_DB_NAME || "traffic_counter";
    if (!uri) return;
    const client = new MongoClient(uri);
    await client.connect();
    await client.db(dbName).collection("users").deleteOne({ email: email.toLowerCase() });
    await client.close();
  } catch {
    /* ignore — cleanup is optional */
  }
}

async function main() {
  console.log(`E2E auth test -- frontend=${FRONTEND}`);
  console.log(`  test user: ${EMAIL}`);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  const consoleErrors = [];
  page.on("pageerror", (err) => consoleErrors.push(`PAGE ERROR: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(`CONSOLE: ${msg.text()}`);
  });

  try {
    // ─── 1. Open register page ─────────────────────────────────────────────
    await page.goto(`${FRONTEND}/register`, { waitUntil: "domcontentloaded" });
    record("Register page loads", page.url().endsWith("/register"), `url=${page.url()}`);

    // Title should appear
    const heading = await page.textContent("h1");
    record("Register title 'Buat Akun Baru'", heading?.includes("Buat Akun"), `got: ${heading}`);

    // ─── 2. Fill and submit register form ──────────────────────────────────
    await page.fill('input[id="name"]', NAME);
    await page.fill('input[id="email"]', EMAIL);
    await page.fill('input[id="password"]', PASSWORD);
    await page.fill('input[id="confirm"]', PASSWORD);

    // Listen for the actual register network request to confirm it goes to /api
    const reqPromise = page.waitForRequest(
      (req) => req.method() === "POST" && req.url().includes("/auth/register"),
      { timeout: 10000 },
    );

    await page.click('button[type="submit"]');
    let registerReq;
    try {
      registerReq = await reqPromise;
      record("POST /auth/register fired by browser", true, `url=${registerReq.url()}`);
    } catch (err) {
      record("POST /auth/register fired by browser", false, `timeout: ${err.message}`);
    }

    // ─── 3. Confirm register success: redirect to /dashboard ───────────────
    try {
      await page.waitForURL(/\/dashboard$/, { timeout: 10000 });
      record("Register redirects to /dashboard", true);
    } catch (err) {
      // Capture any visible error message on the page for diagnostics
      const errBox = await page.$(".text-red-600, .text-red-400");
      const errText = errBox ? (await errBox.textContent()) : "(no error box)";
      record("Register redirects to /dashboard", false, `still at ${page.url()} | error shown: ${errText?.trim()}`);
      throw new Error("register failed; aborting login phase");
    }

    // ─── 4. Verify localStorage has tokens ─────────────────────────────────
    const tokens = await page.evaluate(() => ({
      access: !!localStorage.getItem("traffic_access_token"),
      refresh: !!localStorage.getItem("traffic_refresh_token"),
      user: !!localStorage.getItem("traffic_user"),
    }));
    record("Access token persisted to localStorage", tokens.access);
    record("Refresh token persisted to localStorage", tokens.refresh);
    record("User profile persisted to localStorage", tokens.user);

    // ─── 5. Logout (clear), then go to /login ──────────────────────────────
    await page.evaluate(() => {
      localStorage.removeItem("traffic_access_token");
      localStorage.removeItem("traffic_refresh_token");
      localStorage.removeItem("traffic_user");
    });
    await page.goto(`${FRONTEND}/login`, { waitUntil: "domcontentloaded" });
    record("Login page loads", page.url().endsWith("/login"), `url=${page.url()}`);

    // ─── 6. Login with the registered creds ────────────────────────────────
    await page.fill('input[id="email"]', EMAIL);
    await page.fill('input[id="password"]', PASSWORD);
    await page.click('button[type="submit"]');
    try {
      await page.waitForURL(/\/dashboard$/, { timeout: 10000 });
      record("Login redirects to /dashboard", true);
    } catch (err) {
      const errBox = await page.$(".text-red-600, .text-red-400");
      const errText = errBox ? (await errBox.textContent()) : "(no error box)";
      record("Login redirects to /dashboard", false, `still at ${page.url()} | error: ${errText?.trim()}`);
    }

    // ─── 7. Negative case: wrong password → friendly Indonesian message ────
    await page.evaluate(() => {
      localStorage.removeItem("traffic_access_token");
      localStorage.removeItem("traffic_refresh_token");
      localStorage.removeItem("traffic_user");
    });
    await page.goto(`${FRONTEND}/login`, { waitUntil: "domcontentloaded" });
    await page.fill('input[id="email"]', EMAIL);
    await page.fill('input[id="password"]', "WRONG_PASSWORD");
    await page.click('button[type="submit"]');

    // Wait for an error box to appear
    let errMsg = "";
    try {
      await page.waitForSelector(".text-red-600, .text-red-400", { timeout: 5000 });
      errMsg = (await page.textContent(".text-red-600, .text-red-400")) || "";
    } catch {
      errMsg = "(no error shown)";
    }
    record(
      "Wrong password shows friendly Indonesian error (no 'Failed to fetch')",
      errMsg.includes("salah") || errMsg.includes("Email"),
      `err='${errMsg.trim()}'`,
    );
    record(
      "Error message NOT raw 'Failed to fetch'",
      !errMsg.toLowerCase().includes("failed to fetch"),
      `err='${errMsg.trim()}'`,
    );

    // ─── 8. Negative case: register duplicate email → friendly message ─────
    await page.goto(`${FRONTEND}/register`, { waitUntil: "domcontentloaded" });
    await page.fill('input[id="name"]', NAME);
    await page.fill('input[id="email"]', EMAIL); // already registered above
    await page.fill('input[id="password"]', PASSWORD);
    await page.fill('input[id="confirm"]', PASSWORD);
    await page.click('button[type="submit"]');
    try {
      await page.waitForSelector(".text-red-600, .text-red-400", { timeout: 5000 });
      errMsg = (await page.textContent(".text-red-600, .text-red-400")) || "";
    } catch {
      errMsg = "(no error)";
    }
    record(
      "Duplicate email shows friendly message",
      errMsg.toLowerCase().includes("email") || errMsg.includes("terdaftar"),
      `err='${errMsg.trim()}'`,
    );

    // ─── 9. Frontend-side validation: password mismatch ────────────────────
    await page.goto(`${FRONTEND}/register`, { waitUntil: "domcontentloaded" });
    await page.fill('input[id="name"]', "Mismatch");
    await page.fill('input[id="email"]', `mismatch_${RND}@t.l`);
    await page.fill('input[id="password"]', "abc12345");
    await page.fill('input[id="confirm"]', "different");
    await page.click('button[type="submit"]');
    try {
      await page.waitForSelector(".text-red-600, .text-red-400", { timeout: 3000 });
      errMsg = (await page.textContent(".text-red-600, .text-red-400")) || "";
    } catch {
      errMsg = "";
    }
    record(
      "Password mismatch shows 'tidak cocok'",
      errMsg.toLowerCase().includes("cocok") || errMsg.toLowerCase().includes("konfirmasi"),
      `err='${errMsg.trim()}'`,
    );

  } catch (e) {
    console.log(`\nABORTED: ${e.message}`);
  } finally {
    await context.close();
    await browser.close();
  }

  if (consoleErrors.length) {
    console.log("\nConsole errors captured during test:");
    consoleErrors.slice(0, 10).forEach((e) => console.log(`  ${e}`));
  }

  await cleanupUser(EMAIL);
  await cleanupUser(`mismatch_${RND}@t.l`);

  const passed = results.filter((r) => r.ok).length;
  const failed = results.length - passed;
  console.log("\n" + "=".repeat(70));
  console.log(`SUMMARY: ${passed} passed, ${failed} failed (of ${results.length} total)`);
  if (failed) {
    console.log("\nFAILURES:");
    results.filter((r) => !r.ok).forEach((r) => console.log(`  - ${r.name}: ${r.detail}`));
  }
  console.log("=".repeat(70));
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error("E2E runner crashed:", e);
  process.exit(2);
});
