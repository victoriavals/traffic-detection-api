"""
QA End-to-End Test -- Phase A (Branch CRUD) + Phase B (Endpoint Gating).

Run after backend is up at http://127.0.0.1:3219.

    python tests/qa_phase_a_b.py

Strategy:
- Use unique test emails (qa_admin_<rnd>@test.local, qa_op_<rnd>@test.local) so
  we don't pollute the real users collection. The first registered user becomes
  admin in this DB only if there are 0 users -- otherwise the test admin will
  actually be an operator, breaking admin-only tests. We auto-detect this and
  promote the test admin via direct DB write if needed.
- Branch records use a random suffix to avoid collisions across runs; cleaned
  up at the end.
- Each test prints PASS/FAIL with an explanation.
- Exit code = number of failed tests (0 = all green).
"""

import asyncio
import os
import secrets
import sys

import httpx
import websockets

# Ensure project root is importable so we can flip the test admin's role via
# the same Mongo connection the app uses.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = os.environ.get("QA_BASE", "http://127.0.0.1:3219")
RND = secrets.token_hex(4)

ADMIN_EMAIL = f"qa_admin_{RND}@test.local"
ADMIN_PASS = "Admin12345"
OP_EMAIL = f"qa_op_{RND}@test.local"
OP_PASS = "Operator12345"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}{(' -- ' + detail) if detail else ''}")
    results.append((name, ok, detail))


# ─── Helpers ────────────────────────────────────────────────────────────────

async def _ensure_admin(client: httpx.AsyncClient, email: str, password: str) -> str:
    """Register a user and ensure their role is admin (promote via DB if needed).
    Returns the access token of the (now-admin) user."""
    r = await client.post(f"{BASE}/auth/register", json={"email": email, "name": "QA Admin", "password": password})
    if r.status_code not in (201, 409):
        raise RuntimeError(f"register failed: {r.status_code} {r.text}")
    if r.status_code == 201 and r.json()["user"]["role"] == "admin":
        return r.json()["access_token"]

    # Either email exists already or new user got 'operator' role (DB had users).
    # Promote via direct DB write -- the same MongoDB the app uses.
    from motor.motor_asyncio import AsyncIOMotorClient
    from constant_var import MONGO_URI, MONGO_DB_NAME
    mc = AsyncIOMotorClient(MONGO_URI)
    await mc[MONGO_DB_NAME].users.update_one({"email": email.lower()}, {"$set": {"role": "admin"}})
    mc.close()

    # Login fresh to pick up new role
    r = await client.post(f"{BASE}/auth/login", json={"email": email, "password": password})
    if r.status_code != 200:
        raise RuntimeError(f"login failed: {r.status_code} {r.text}")
    return r.json()["access_token"]


async def _login(client: httpx.AsyncClient, email: str, password: str) -> str:
    r = await client.post(f"{BASE}/auth/login", json={"email": email, "password": password})
    if r.status_code != 200:
        raise RuntimeError(f"login failed: {r.status_code} {r.text}")
    return r.json()["access_token"]


# ─── Test groups ─────────────────────────────────────────────────────────────

async def test_auth_basic(client: httpx.AsyncClient, admin_tok: str, op_tok: str) -> None:
    print("\n=== AUTH BASIC ===")

    # /auth/me with admin token
    r = await client.get(f"{BASE}/auth/me", headers={"Authorization": f"Bearer {admin_tok}"})
    record("auth/me with admin token returns admin role",
           r.status_code == 200 and r.json().get("role") == "admin",
           f"got status={r.status_code} role={r.json().get('role') if r.status_code==200 else 'n/a'}")

    # /auth/me with operator token
    r = await client.get(f"{BASE}/auth/me", headers={"Authorization": f"Bearer {op_tok}"})
    record("auth/me with operator token returns operator role",
           r.status_code == 200 and r.json().get("role") == "operator",
           f"got status={r.status_code} role={r.json().get('role') if r.status_code==200 else 'n/a'}")

    # /auth/me without token
    r = await client.get(f"{BASE}/auth/me")
    record("auth/me without token -> 401", r.status_code == 401, f"got {r.status_code}")

    # Invalid login
    r = await client.post(f"{BASE}/auth/login", json={"email": ADMIN_EMAIL, "password": "wrongpass"})
    record("login with wrong password -> 401", r.status_code == 401, f"got {r.status_code}")

    # Refresh flow
    r = await client.post(f"{BASE}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASS})
    refresh_tok = r.json()["refresh_token"]
    r = await client.post(f"{BASE}/auth/refresh", json={"refresh_token": refresh_tok})
    record("refresh with valid refresh_token -> 200 + access_token",
           r.status_code == 200 and "access_token" in r.json(),
           f"got {r.status_code}")

    # Refresh with access token (wrong type)
    r = await client.post(f"{BASE}/auth/refresh", json={"refresh_token": admin_tok})
    record("refresh with access_token (wrong type) -> 401", r.status_code == 401, f"got {r.status_code}")

    # Refresh with garbage
    r = await client.post(f"{BASE}/auth/refresh", json={"refresh_token": "not.a.jwt"})
    record("refresh with invalid token -> 401", r.status_code == 401, f"got {r.status_code}")


async def test_branches_phase_a(client: httpx.AsyncClient, admin_tok: str, op_tok: str) -> list[str]:
    """Returns list of created branch IDs for cleanup."""
    print("\n=== PHASE A -- BRANCH CRUD ===")
    A = {"Authorization": f"Bearer {admin_tok}"}
    O = {"Authorization": f"Bearer {op_tok}"}
    created_ids: list[str] = []

    # GET /branches without auth
    r = await client.get(f"{BASE}/branches")
    record("GET /branches without auth -> 401", r.status_code == 401, f"got {r.status_code}")

    # GET /branches with operator (should be allowed -- read open to all auth users)
    r = await client.get(f"{BASE}/branches", headers=O)
    record("GET /branches with operator token -> 200 (read open to authed users)",
           r.status_code == 200 and isinstance(r.json(), list),
           f"got {r.status_code}")

    # POST /branches as operator -> 201 (v1.6.0: all authed users can CRUD)
    op_branch_name = f"qa-op-branch-{RND}"
    r = await client.post(f"{BASE}/branches", json={"name": op_branch_name, "address": "Jl. Operator"}, headers=O)
    op_ok = r.status_code == 201 and r.json().get("created_by")
    record("POST /branches as operator -> 201 + created_by recorded",
           op_ok,
           f"got {r.status_code} created_by={r.json().get('created_by') if r.status_code == 201 else 'n/a'}")
    if op_ok:
        created_ids.append(r.json()["id"])

    # POST /branches as admin -> 201
    name1 = f"qa-branch-1-{RND}"
    r = await client.post(f"{BASE}/branches", json={"name": name1, "address": "Jl. Sudirman 1"}, headers=A)
    ok = r.status_code == 201 and r.json().get("name") == name1 and r.json().get("created_by") == ADMIN_EMAIL
    record("POST /branches as admin -> 201 + body matches + created_by=admin", ok, f"got {r.status_code}")
    if ok:
        created_ids.append(r.json()["id"])
        bid1 = r.json()["id"]

    # POST duplicate name -> 409
    r = await client.post(f"{BASE}/branches", json={"name": name1, "address": "duplicate"}, headers=A)
    record("POST duplicate branch name -> 409", r.status_code == 409, f"got {r.status_code}")

    # POST empty name -> 422
    r = await client.post(f"{BASE}/branches", json={"name": "", "address": "x"}, headers=A)
    record("POST empty name -> 422", r.status_code == 422, f"got {r.status_code}")

    # POST whitespace-only name -> 422 (strips to empty; backend must reject)
    r = await client.post(f"{BASE}/branches", json={"name": "   ", "address": "x"}, headers=A)
    record("POST whitespace-only name -> 422", r.status_code == 422, f"got {r.status_code}")

    # POST too-long name -> 422
    r = await client.post(f"{BASE}/branches", json={"name": "x" * 201, "address": "x"}, headers=A)
    record("POST name >200 chars -> 422", r.status_code == 422, f"got {r.status_code}")

    # POST name with leading/trailing whitespace -> 201, stored trimmed
    padded_name = f"  qa-padded-{RND}  "
    r = await client.post(f"{BASE}/branches", json={"name": padded_name, "address": ""}, headers=A)
    trimmed_name = padded_name.strip()
    padded_ok = r.status_code == 201 and r.json().get("name") == trimmed_name
    record("POST name with surrounding spaces -> 201, name stored trimmed",
           padded_ok, f"got {r.status_code} name={r.json().get('name') if r.status_code == 201 else 'n/a'}")
    if padded_ok:
        created_ids.append(r.json()["id"])

    # GET specific branch (admin)
    if created_ids:
        r = await client.get(f"{BASE}/branches/{bid1}", headers=A)
        record("GET /branches/{id} -> 200", r.status_code == 200 and r.json()["name"] == name1, f"got {r.status_code}")

    # GET with invalid id format -> 400
    r = await client.get(f"{BASE}/branches/not-an-id", headers=A)
    record("GET /branches/{invalid_id} -> 400", r.status_code == 400, f"got {r.status_code}")

    # GET with valid format but non-existent -> 404
    r = await client.get(f"{BASE}/branches/000000000000000000000000", headers=A)
    record("GET /branches/{nonexistent_id} -> 404", r.status_code == 404, f"got {r.status_code}")

    # PATCH as operator -> 200 (v1.6.0: all authed users can PATCH; updated_by recorded)
    if created_ids:
        r = await client.patch(f"{BASE}/branches/{bid1}", json={"address": "Jl. Operator Edit"}, headers=O)
        ok = (
            r.status_code == 200
            and r.json().get("address") == "Jl. Operator Edit"
            and r.json().get("updated_by") == OP_EMAIL
            and r.json().get("updated_at") is not None
        )
        record("PATCH /branches/{id} as operator -> 200 + updated_by=operator + timestamp set",
               ok, f"got {r.status_code} updated_by={r.json().get('updated_by') if r.status_code == 200 else 'n/a'}")

        # PATCH as admin -> 200 + updated_by switches to admin
        r = await client.patch(f"{BASE}/branches/{bid1}", json={"address": "Jl. Updated"}, headers=A)
        ok = (
            r.status_code == 200
            and r.json().get("address") == "Jl. Updated"
            and r.json().get("updated_by") == ADMIN_EMAIL
        )
        record("PATCH /branches/{id} as admin -> 200 + updated_by=admin",
               ok, f"got {r.status_code}")

        # PATCH with whitespace-only name -> 422
        r = await client.patch(f"{BASE}/branches/{bid1}", json={"name": "   "}, headers=A)
        record("PATCH whitespace-only name -> 422", r.status_code == 422, f"got {r.status_code}")

        # PATCH rename to name of another existing branch -> 409 (requires op branch to exist)
        if op_ok:
            r = await client.patch(f"{BASE}/branches/{bid1}", json={"name": op_branch_name}, headers=A)
            record("PATCH rename to existing branch name -> 409", r.status_code == 409, f"got {r.status_code}")
        else:
            record("PATCH rename to existing branch name -> 409", False, "skipped: op branch not created")

        # PATCH empty body -> 400
        r = await client.patch(f"{BASE}/branches/{bid1}", json={}, headers=A)
        record("PATCH with empty body -> 400", r.status_code == 400, f"got {r.status_code}")

    # Create another branch then delete
    name2 = f"qa-branch-2-{RND}"
    r = await client.post(f"{BASE}/branches", json={"name": name2, "address": "x"}, headers=A)
    if r.status_code == 201:
        bid2 = r.json()["id"]
        created_ids.append(bid2)

        # DELETE as operator -> 204 (v1.6.0: all authed users can DELETE)
        r = await client.delete(f"{BASE}/branches/{bid2}", headers=O)
        record("DELETE /branches/{id} as operator -> 204", r.status_code == 204, f"got {r.status_code}")
        if r.status_code == 204:
            created_ids.remove(bid2)

        # Create another branch and let admin delete this time
        name3 = f"qa-branch-3-{RND}"
        r = await client.post(f"{BASE}/branches", json={"name": name3, "address": "x"}, headers=A)
        if r.status_code == 201:
            bid3 = r.json()["id"]
            created_ids.append(bid3)
            r = await client.delete(f"{BASE}/branches/{bid3}", headers=A)
            record("DELETE /branches/{id} as admin -> 204", r.status_code == 204, f"got {r.status_code}")
            if r.status_code == 204:
                created_ids.remove(bid3)

        # DELETE non-existent -> 404
        r = await client.delete(f"{BASE}/branches/000000000000000000000000", headers=A)
        record("DELETE /branches/{nonexistent_id} -> 404", r.status_code == 404, f"got {r.status_code}")

    return created_ids


async def test_endpoint_gating_phase_b(client: httpx.AsyncClient, admin_tok: str, _op_tok: str) -> None:
    print("\n=== PHASE B -- HTTP ENDPOINT GATING ===")
    A = {"Authorization": f"Bearer {admin_tok}"}

    # Sample of endpoints across all gated route modules
    samples = [
        ("GET", "/logs", None),
        ("GET", "/stats/summary", None),
        ("GET", "/stats/hourly", None),
        ("GET", "/stats/hourly-detections", None),
        ("GET", "/stats/weekly-detections", None),
        ("GET", "/stats/report?date_from=2026-01-01&date_to=2026-01-31&hour_from=0&hour_to=23", None),
        ("GET", "/video/jobs", None),
        ("GET", "/ezviz/status", None),
    ]
    for method, path, body in samples:
        # Without token -> 401
        r = await client.request(method, f"{BASE}{path}", json=body)
        record(f"{method} {path[:40]} without token -> 401", r.status_code == 401, f"got {r.status_code}")

    # With token -> not 401 (could be 200 / 503 if Mongo issue / 404 etc -- just NOT 401)
    for method, path, body in samples:
        r = await client.request(method, f"{BASE}{path}", json=body, headers=A)
        record(f"{method} {path[:40]} with token -> not 401", r.status_code != 401, f"got {r.status_code}")

    # Public endpoint -- / should still work without token
    r = await client.get(f"{BASE}/")
    record("GET / (health) without token -> 200 (public)", r.status_code == 200, f"got {r.status_code}")


async def test_websocket_auth_phase_b(admin_tok: str) -> None:
    print("\n=== PHASE B -- WEBSOCKET AUTH ===")
    base_ws = BASE.replace("http://", "ws://")

    # WS without token -> close 4401
    try:
        async with websockets.connect(f"{base_ws}/rtsp/stream", open_timeout=5) as ws:
            await ws.recv()
            record("WS /rtsp/stream without token -> close 4401", False, "connection accepted unexpectedly")
    except websockets.exceptions.InvalidStatus as e:
        # Fastapi rejects with HTTP error before upgrade if Query is required
        record("WS /rtsp/stream without token -> connection rejected", True, f"http {e.response.status_code}")
    except websockets.exceptions.ConnectionClosed as e:
        record("WS /rtsp/stream without token -> close 4401",
               e.code == 4401, f"close code={e.code} reason={e.reason!r}")
    except Exception as e:
        record("WS /rtsp/stream without token -> close 4401", False, f"unexpected: {type(e).__name__}: {e}")

    # WS with invalid token -> close 4401
    try:
        async with websockets.connect(f"{base_ws}/rtsp/stream?token=invalid.jwt.here", open_timeout=5) as ws:
            await ws.recv()
            record("WS /rtsp/stream with bad token -> close 4401", False, "connection accepted unexpectedly")
    except websockets.exceptions.InvalidStatus as e:
        record("WS /rtsp/stream with bad token -> connection rejected", True, f"http {e.response.status_code}")
    except websockets.exceptions.ConnectionClosed as e:
        record("WS /rtsp/stream with bad token -> close 4401",
               e.code == 4401, f"close code={e.code}")
    except Exception as e:
        record("WS /rtsp/stream with bad token -> close 4401", False, f"unexpected: {type(e).__name__}: {e}")

    # WS with valid token -> accept (we just verify accept happens, then close)
    try:
        async with websockets.connect(f"{base_ws}/rtsp/stream?token={admin_tok}", open_timeout=5) as ws:
            # Server will wait for config message; we just verify the upgrade succeeded
            record("WS /rtsp/stream with valid token -> accepted", True, "connection opened")
            await ws.close()
    except Exception as e:
        record("WS /rtsp/stream with valid token -> accepted", False, f"failed: {type(e).__name__}: {e}")

    # Same for ezviz
    try:
        async with websockets.connect(f"{base_ws}/ezviz/stream", open_timeout=5) as ws:
            await ws.recv()
            record("WS /ezviz/stream without token -> close 4401", False, "connection accepted unexpectedly")
    except websockets.exceptions.InvalidStatus as e:
        record("WS /ezviz/stream without token -> connection rejected", True, f"http {e.response.status_code}")
    except websockets.exceptions.ConnectionClosed as e:
        record("WS /ezviz/stream without token -> close 4401",
               e.code == 4401, f"close code={e.code}")
    except Exception as e:
        record("WS /ezviz/stream without token -> close 4401", False, f"unexpected: {type(e).__name__}: {e}")


# ─── Cleanup ─────────────────────────────────────────────────────────────────

async def cleanup(client: httpx.AsyncClient, admin_tok: str, branch_ids: list[str]) -> None:
    print("\n=== CLEANUP ===")
    A = {"Authorization": f"Bearer {admin_tok}"}
    for bid in branch_ids:
        r = await client.delete(f"{BASE}/branches/{bid}", headers=A)
        print(f"  removed branch {bid}: {r.status_code}")

    # Remove test users via direct DB write
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from constant_var import MONGO_URI, MONGO_DB_NAME
        mc = AsyncIOMotorClient(MONGO_URI)
        res = await mc[MONGO_DB_NAME].users.delete_many({"email": {"$in": [ADMIN_EMAIL, OP_EMAIL]}})
        mc.close()
        print(f"  removed {res.deleted_count} test users")
    except Exception as e:
        print(f"  user cleanup failed: {e}")


# ─── Main ────────────────────────────────────────────────────────────────────

async def main() -> int:
    print(f"QA test run -- base={BASE} rnd={RND}")
    print(f"  admin: {ADMIN_EMAIL}")
    print(f"  operator: {OP_EMAIL}")

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Bootstrap users
        admin_tok = await _ensure_admin(client, ADMIN_EMAIL, ADMIN_PASS)

        r = await client.post(f"{BASE}/auth/register", json={"email": OP_EMAIL, "name": "QA Operator", "password": OP_PASS})
        if r.status_code == 201:
            op_tok = r.json()["access_token"]
        elif r.status_code == 409:
            op_tok = await _login(client, OP_EMAIL, OP_PASS)
        else:
            raise RuntimeError(f"operator register failed: {r.status_code} {r.text}")

        # Run test groups
        await test_auth_basic(client, admin_tok, op_tok)
        branch_ids = await test_branches_phase_a(client, admin_tok, op_tok)
        await test_endpoint_gating_phase_b(client, admin_tok, op_tok)
        await test_websocket_auth_phase_b(admin_tok)

        # Cleanup
        await cleanup(client, admin_tok, branch_ids)

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed} passed, {failed} failed (of {len(results)} total)")
    if failed:
        print("\nFAILURES:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
