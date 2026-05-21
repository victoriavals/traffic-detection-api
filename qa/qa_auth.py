"""
QA End-to-End Test -- AUTH (Register + Login + Refresh + Me).

Comprehensive test of register & login features:
- Positive flow (register -> login -> refresh -> me)
- Negative cases (validation, wrong creds, duplicates)
- Edge cases (whitespace, case sensitivity, unicode, very long)
- Security (SQL injection, XSS, password not leaked, JWT integrity)

Run after backend is up. Set BASE env to override default URL.

    uv run python qa/qa_auth.py

Each test prints PASS/FAIL. Exit code = number of failed tests.
"""

import asyncio
import os
import secrets
import sys

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = os.environ.get("QA_BASE", "http://192.168.1.8:3219")
RND = secrets.token_hex(4)

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}{(' -- ' + detail) if detail else ''}")
    results.append((name, ok, detail))


# --- Positive: Register ------------------------------------------------------

async def test_register_positive(client: httpx.AsyncClient) -> dict:
    """Returns dict of registered users for later cleanup + reuse."""
    print("\n=== REGISTER -- POSITIVE ===")
    users: dict[str, dict] = {}

    email = f"qa_pos_{RND}@test.local"
    r = await client.post(f"{BASE}/auth/register", json={
        "email": email, "name": "QA Pos", "password": "Pass12345"
    })
    ok = (
        r.status_code == 201
        and "access_token" in r.json()
        and "refresh_token" in r.json()
        and r.json()["user"]["email"] == email
        and r.json()["user"]["role"] in ("admin", "operator")
        and "password" not in r.text
        and "password_hash" not in r.text
    )
    record("Register valid user -> 201 + tokens + user; no password leaked", ok,
           f"got {r.status_code}")
    if r.status_code == 201:
        users["main"] = {"email": email, "password": "Pass12345",
                         "token": r.json()["access_token"],
                         "refresh": r.json()["refresh_token"],
                         "role": r.json()["user"]["role"]}

    # Email is stored lowercase: register Mixed.Case email then login lowercase
    mixed_email = f"QA_Mix_{RND}@TEST.local"
    r = await client.post(f"{BASE}/auth/register", json={
        "email": mixed_email, "name": "QA Mix", "password": "Pass12345"
    })
    stored_email = r.json().get("user", {}).get("email") if r.status_code == 201 else None
    record("Register mixed-case email -> stored lowercase", stored_email == mixed_email.lower(),
           f"got status={r.status_code} stored={stored_email}")
    if r.status_code == 201:
        users["mix"] = {"email": mixed_email, "password": "Pass12345",
                        "token": r.json()["access_token"]}

    return users


# --- Negative: Register validation -------------------------------------------

async def test_register_negative(client: httpx.AsyncClient) -> None:
    print("\n=== REGISTER -- NEGATIVE & VALIDATION ===")

    # Duplicate email -> 409
    email = f"qa_dup_{RND}@test.local"
    r1 = await client.post(f"{BASE}/auth/register", json={
        "email": email, "name": "First", "password": "Pass12345"
    })
    r2 = await client.post(f"{BASE}/auth/register", json={
        "email": email, "name": "Second", "password": "Pass12345"
    })
    record("Register duplicate email -> 409", r2.status_code == 409,
           f"first={r1.status_code} dup={r2.status_code}")

    # Duplicate email different case -> 409 (because we normalize to lowercase)
    email_upper = email.upper()
    r3 = await client.post(f"{BASE}/auth/register", json={
        "email": email_upper, "name": "Dup", "password": "Pass12345"
    })
    record("Register duplicate email (case-different) -> 409",
           r3.status_code == 409, f"got {r3.status_code}")

    cases = [
        ("invalid email format (no @)", {"email": "noatsign", "name": "X", "password": "Pass12345"}, 422),
        ("invalid email format (no domain)", {"email": "x@", "name": "X", "password": "Pass12345"}, 422),
        ("invalid email format (spaces)", {"email": "a b@c.com", "name": "X", "password": "Pass12345"}, 422),
        ("empty email", {"email": "", "name": "X", "password": "Pass12345"}, 422),
        ("empty name", {"email": f"qa_en_{RND}@t.l", "name": "", "password": "Pass12345"}, 422),
        ("empty password", {"email": f"qa_ep_{RND}@t.l", "name": "X", "password": ""}, 422),
        ("password < 6 chars", {"email": f"qa_pw_{RND}@t.l", "name": "X", "password": "12345"}, 422),
        ("password > 128 chars", {"email": f"qa_pl_{RND}@t.l", "name": "X", "password": "a" * 129}, 422),
        ("name > 100 chars", {"email": f"qa_nl_{RND}@t.l", "name": "x" * 101, "password": "Pass12345"}, 422),
        ("email > 200 chars", {"email": "a" * 200 + "@t.l", "name": "X", "password": "Pass12345"}, 422),
        ("missing email field", {"name": "X", "password": "Pass12345"}, 422),
        ("missing name field", {"email": f"qa_mn_{RND}@t.l", "password": "Pass12345"}, 422),
        ("missing password field", {"email": f"qa_mp_{RND}@t.l", "name": "X"}, 422),
    ]
    for name, payload, expected in cases:
        r = await client.post(f"{BASE}/auth/register", json=payload)
        record(f"Register {name} -> {expected}", r.status_code == expected, f"got {r.status_code}")

    # Whitespace-only name (length 3 passes min_length=1, but is essentially empty)
    r = await client.post(f"{BASE}/auth/register", json={
        "email": f"qa_ws_{RND}@t.l", "name": "   ", "password": "Pass12345"
    })
    # Expected: 422 (post-strip validation should catch this, like we did for branches)
    record("Register whitespace-only name -> 422 (defensive validation)",
           r.status_code == 422, f"got {r.status_code}")

    # Email with leading/trailing whitespace -- pattern requires no whitespace
    r = await client.post(f"{BASE}/auth/register", json={
        "email": f" qa_es_{RND}@t.l ", "name": "X", "password": "Pass12345"
    })
    record("Register email with leading/trailing whitespace -> 422",
           r.status_code == 422, f"got {r.status_code}")

    # Empty JSON body
    r = await client.post(f"{BASE}/auth/register", json={})
    record("Register empty body -> 422", r.status_code == 422, f"got {r.status_code}")

    # Malformed JSON
    r = await client.post(f"{BASE}/auth/register",
                          content="not json", headers={"Content-Type": "application/json"})
    record("Register malformed JSON -> 422", r.status_code == 422, f"got {r.status_code}")

    # Wrong content type
    r = await client.post(f"{BASE}/auth/register",
                          content="x=1", headers={"Content-Type": "application/x-www-form-urlencoded"})
    record("Register form-urlencoded body -> 422", r.status_code == 422, f"got {r.status_code}")


# --- Login ------------------------------------------------------------------

async def test_login(client: httpx.AsyncClient, users: dict) -> dict:
    print("\n=== LOGIN ===")
    fresh: dict[str, dict] = {}

    if "main" not in users:
        record("Login: main user available for testing", False, "skipped -- register failed")
        return fresh

    cred = users["main"]

    # Valid login -> 200
    r = await client.post(f"{BASE}/auth/login", json={
        "email": cred["email"], "password": cred["password"]
    })
    ok = (
        r.status_code == 200
        and "access_token" in r.json()
        and "refresh_token" in r.json()
        and r.json()["user"]["email"] == cred["email"]
        and "password" not in r.text and "password_hash" not in r.text
    )
    record("Login valid creds -> 200 + tokens + no password leak", ok,
           f"got {r.status_code}")
    if r.status_code == 200:
        fresh["main"] = {**cred, "token": r.json()["access_token"], "refresh": r.json()["refresh_token"]}

    # Login with mixed-case email (registered lowercase, login UPPER) -> 200
    if "mix" in users:
        r = await client.post(f"{BASE}/auth/login", json={
            "email": users["mix"]["email"].upper(), "password": users["mix"]["password"]
        })
        record("Login email case-insensitive -> 200", r.status_code == 200, f"got {r.status_code}")

    # Wrong password -> 401
    r = await client.post(f"{BASE}/auth/login", json={
        "email": cred["email"], "password": "WRONG_PASSWORD"
    })
    record("Login wrong password -> 401", r.status_code == 401, f"got {r.status_code}")

    # Non-existent user -> 401 (must NOT be 404 -- that would leak account existence)
    r = await client.post(f"{BASE}/auth/login", json={
        "email": f"ghost_{RND}@test.local", "password": "anything"
    })
    record("Login non-existent email -> 401 (no enumeration leak)",
           r.status_code == 401, f"got {r.status_code}")

    # Validation errors
    cases = [
        ("missing email", {"password": "x"}, 422),
        ("missing password", {"email": cred["email"]}, 422),
        ("empty body", {}, 422),
    ]
    for name, payload, expected in cases:
        r = await client.post(f"{BASE}/auth/login", json=payload)
        record(f"Login {name} -> {expected}", r.status_code == expected, f"got {r.status_code}")

    # SQL/NoSQL injection: $ne operator should NOT bypass auth
    # If Pydantic accepts dict as email, MongoDB would interpret {"$ne": null} as match-anything.
    r = await client.post(f"{BASE}/auth/login", json={
        "email": {"$ne": None}, "password": {"$ne": None}
    })
    record("Login NoSQL injection ($ne) -> blocked (422 or 401)",
           r.status_code in (401, 422), f"got {r.status_code}")

    # XSS-style payload in email field -- regex is permissive (only checks @ + .),
    # so it passes validation; non-existent user returns 401 (correct -- no enumeration leak).
    # Frontend renders all user-controlled text via React (auto-escaped), so XSS payload
    # in email field is not executable downstream.
    r = await client.post(f"{BASE}/auth/login", json={
        "email": "<script>alert(1)</script>@x.com", "password": "x"
    })
    record("Login XSS-ish email -> 401 (not 200; no auth bypass)",
           r.status_code == 401, f"got {r.status_code}")

    return fresh


# --- Refresh ----------------------------------------------------------------

async def test_refresh(client: httpx.AsyncClient, fresh: dict) -> None:
    print("\n=== REFRESH ===")
    if "main" not in fresh:
        record("Refresh: main user token available", False, "skipped")
        return

    tok = fresh["main"]
    # Valid refresh -> 200
    r = await client.post(f"{BASE}/auth/refresh", json={"refresh_token": tok["refresh"]})
    record("Refresh valid -> 200 + new access_token",
           r.status_code == 200 and "access_token" in r.json(),
           f"got {r.status_code}")

    # Use access_token as refresh -> 401
    r = await client.post(f"{BASE}/auth/refresh", json={"refresh_token": tok["token"]})
    record("Refresh with access_token (wrong type) -> 401", r.status_code == 401,
           f"got {r.status_code}")

    # Garbage token -> 401
    r = await client.post(f"{BASE}/auth/refresh", json={"refresh_token": "not.a.jwt"})
    record("Refresh with garbage token -> 401", r.status_code == 401, f"got {r.status_code}")

    # Empty refresh_token field -> 401 or 422 (both acceptable depending on validation)
    r = await client.post(f"{BASE}/auth/refresh", json={"refresh_token": ""})
    record("Refresh with empty string -> 401/422",
           r.status_code in (401, 422), f"got {r.status_code}")

    # Missing refresh_token field -> 422
    r = await client.post(f"{BASE}/auth/refresh", json={})
    record("Refresh missing field -> 422", r.status_code == 422, f"got {r.status_code}")

    # Tampered token (last char flipped) -> 401
    bad = tok["refresh"][:-1] + ("A" if tok["refresh"][-1] != "A" else "B")
    r = await client.post(f"{BASE}/auth/refresh", json={"refresh_token": bad})
    record("Refresh with tampered token -> 401", r.status_code == 401, f"got {r.status_code}")


# --- /auth/me ---------------------------------------------------------------

async def test_me(client: httpx.AsyncClient, fresh: dict) -> None:
    print("\n=== /auth/me ===")
    if "main" not in fresh:
        record("/auth/me: main token available", False, "skipped")
        return

    tok = fresh["main"]["token"]

    # No token -> 401
    r = await client.get(f"{BASE}/auth/me")
    record("/auth/me without token -> 401", r.status_code == 401, f"got {r.status_code}")

    # Valid token -> 200
    r = await client.get(f"{BASE}/auth/me", headers={"Authorization": f"Bearer {tok}"})
    ok = (
        r.status_code == 200
        and r.json()["email"] == fresh["main"]["email"]
        and "password_hash" not in r.text
    )
    record("/auth/me valid token -> 200 + user + no password_hash leak", ok,
           f"got {r.status_code}")

    # Bearer prefix missing -> 401
    r = await client.get(f"{BASE}/auth/me", headers={"Authorization": tok})
    record("/auth/me without Bearer prefix -> 401", r.status_code == 401, f"got {r.status_code}")

    # Garbage token -> 401
    r = await client.get(f"{BASE}/auth/me", headers={"Authorization": "Bearer garbage.token.here"})
    record("/auth/me garbage token -> 401", r.status_code == 401, f"got {r.status_code}")

    # Tampered token -> 401
    bad = tok[:-1] + ("A" if tok[-1] != "A" else "B")
    r = await client.get(f"{BASE}/auth/me", headers={"Authorization": f"Bearer {bad}"})
    record("/auth/me tampered token -> 401", r.status_code == 401, f"got {r.status_code}")


# --- Cleanup ----------------------------------------------------------------

async def cleanup(emails: list[str]) -> None:
    print("\n=== CLEANUP ===")
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from constant_var import MONGO_URI, MONGO_DB_NAME
        mc = AsyncIOMotorClient(MONGO_URI)
        # Lowercase since storage normalizes
        lc = [e.lower() for e in emails]
        res = await mc[MONGO_DB_NAME].users.delete_many({"email": {"$in": lc}})
        mc.close()
        print(f"  removed {res.deleted_count} test users")
    except Exception as e:
        print(f"  cleanup failed: {e}")


# --- Main -------------------------------------------------------------------

async def main() -> int:
    print(f"QA AUTH test -- base={BASE} rnd={RND}")

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Verify backend up
        try:
            r = await client.get(f"{BASE}/")
            if r.status_code != 200:
                print(f"FATAL: backend health check failed (status {r.status_code})")
                return 1
        except Exception as e:
            print(f"FATAL: cannot reach backend at {BASE}: {e}")
            return 1

        users = await test_register_positive(client)
        await test_register_negative(client)
        fresh = await test_login(client, users)
        await test_refresh(client, fresh)
        await test_me(client, fresh)

        # Cleanup
        emails_to_remove = []
        for v in users.values():
            if "email" in v:
                emails_to_remove.append(v["email"])
        # Plus all the ones from negative tests with our RND suffix
        for prefix in ["qa_dup", "qa_en", "qa_ep", "qa_pw", "qa_pl", "qa_nl",
                       "qa_mn", "qa_mp", "qa_ws", "qa_es", "qa_health_check"]:
            emails_to_remove.append(f"{prefix}_{RND}@t.l")
        emails_to_remove.append(f"qa_health_check@test.local")
        await cleanup(emails_to_remove)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passed} passed, {failed} failed (of {len(results)} total)")
    if failed:
        print("\nFAILURES:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 70)
    return failed


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
