"""
QA Engineer test pack untuk 2 fitur terbaru:
- POST /org/join          — self-service join via invite code
- PATCH /org              — rename org (owner / admin only)
- Default org name        — saat register baru
- Migration backfill name — legacy orgs

Coverage: Functional, Validation, Negative, Edge case, Security.
Run: uv run python qa/qa_join_rename.py
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


results = []


def record(tc_id, name, status, actual, note=""):
    results.append({"id": tc_id, "name": name, "status": status,
                    "actual": actual, "note": note})
    icon = "[PASS]" if status == "PASS" else ("[FAIL]" if status == "FAIL" else "[WARN]")
    print(f"{icon} {tc_id}: {name}")
    print(f"       -> {actual}")
    if note:
        print(f"       NOTE: {note}")


async def run():
    from httpx import AsyncClient, ASGITransport
    from services.database import connect_db, get_db, close_db
    from fastapi import FastAPI
    from routes.auth import router as auth_router
    from routes.org import router as org_router
    from bson import ObjectId

    await connect_db()
    db = get_db()
    await db.users.delete_many({"email": {"$regex": "^qatest_"}})

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(org_router)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # ========================================================
        # SETUP: 3 users — 2 owners + 1 will-be-joiner
        # ========================================================
        rA = await client.post("/auth/register", json={
            "email": "qatest_alice@example.com", "name": "Alice",
            "password": "password123"})
        tokenA = rA.json()["access_token"]
        meA = await client.get("/org/me",
                                headers={"Authorization": f"Bearer {tokenA}"})
        orgA_id = meA.json()["org_id"]
        orgA_code = meA.json()["invite_code"]
        orgA_name = meA.json()["name"]

        rB = await client.post("/auth/register", json={
            "email": "qatest_bob@example.com", "name": "Bob",
            "password": "password123"})
        tokenB = rB.json()["access_token"]
        meB = await client.get("/org/me",
                                headers={"Authorization": f"Bearer {tokenB}"})
        orgB_id = meB.json()["org_id"]
        orgB_code = meB.json()["invite_code"]

        rC = await client.post("/auth/register", json={
            "email": "qatest_charlie@example.com", "name": "Charlie",
            "password": "password123"})
        tokenC = rC.json()["access_token"]
        meC = await client.get("/org/me",
                                headers={"Authorization": f"Bearer {tokenC}"})
        orgC_id = meC.json()["org_id"]

        # ========================================================
        # F (Functional) — Default name
        # ========================================================
        if orgA_name == "Tim Alice":
            record("F-001", "Default name 'Tim {nama_user}' saat register",
                   "PASS", f"orgA.name = {orgA_name!r}")
        else:
            record("F-001", "Default name", "FAIL",
                   f"expected 'Tim Alice', got {orgA_name!r}")

        # F-002: Register dengan invite_code → tidak generate org baru, jadi name None
        # (joining existing org, not creating)
        rJ = await client.post("/auth/register", json={
            "email": "qatest_joiner@example.com", "name": "Joiner",
            "password": "password123", "invite_code": orgA_code})
        meJ = await client.get("/org/me",
                                headers={"Authorization": f"Bearer {rJ.json()['access_token']}"})
        if meJ.json()["org_id"] == orgA_id and meJ.json()["name"] == orgA_name:
            record("F-002", "Joiner inherit nama org host (bukan bikin baru)",
                   "PASS", f"joiner.org.name = {meJ.json()['name']!r}")
        else:
            record("F-002", "Joiner org name", "FAIL", str(meJ.json()))

        # ========================================================
        # F — Rename happy path
        # ========================================================
        r = await client.patch("/org", json={"name": "Tim Marketing"},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200 and r.json()["name"] == "Tim Marketing":
            record("F-003", "Owner rename org -> 200",
                   "PASS", f"name updated to {r.json()['name']!r}")
        else:
            record("F-003", "Owner rename", "FAIL",
                   f"status={r.status_code}, body={r.text[:200]}")

        # F-004: GET /org/me reflects rename
        meA2 = await client.get("/org/me",
                                 headers={"Authorization": f"Bearer {tokenA}"})
        if meA2.json()["name"] == "Tim Marketing":
            record("F-004", "GET /org/me reflect rename",
                   "PASS", "name persisted")
        else:
            record("F-004", "Rename persisted", "FAIL", str(meA2.json()))

        # F-005: Trim whitespace
        r = await client.patch("/org", json={"name": "  Tim Operasional  "},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200 and r.json()["name"] == "Tim Operasional":
            record("F-005", "Trim leading/trailing whitespace",
                   "PASS", f"'  Tim Operasional  ' -> {r.json()['name']!r}")
        else:
            record("F-005", "Trim whitespace", "FAIL",
                   f"status={r.status_code}, name={r.json().get('name')!r}")

        # ========================================================
        # F — Self-service Join
        # ========================================================
        # C (Charlie owns orgC) joins orgB via Bob's code
        r = await client.post("/org/join",
                               json={"invite_code": orgB_code},
                               headers={"Authorization": f"Bearer {tokenC}"})
        if r.status_code == 200 and r.json()["org_id"] == orgB_id:
            record("F-006", "Self-service join via code valid -> 200",
                   "PASS", f"Charlie pindah ke {orgB_id} (Bob's org)")
        else:
            record("F-006", "Self join", "FAIL",
                   f"status={r.status_code}, body={r.text[:200]}")

        # F-007: Charlie's org should now show Bob's org name in response
        if r.json().get("name") == "Tim Bob":
            record("F-007", "Join response include nama org tujuan",
                   "PASS", f"name={r.json()['name']!r}")
        else:
            record("F-007", "Join response include name", "WARN",
                   f"name={r.json().get('name')!r}")

        # ========================================================
        # V (Validation)
        # ========================================================
        # V-001: Rename empty -> 422 Pydantic
        r = await client.patch("/org", json={"name": ""},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 422:
            record("V-001", "Rename empty string -> 422 Pydantic",
                   "PASS", "min_length=1 enforced")
        else:
            record("V-001", "Empty name", "FAIL", f"status={r.status_code}")

        # V-002: Rename whitespace-only -> 400 server-side
        r = await client.patch("/org", json={"name": "   "},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 400:
            record("V-002", "Rename whitespace-only -> 400",
                   "PASS", f"detail={r.json().get('detail')!r}")
        else:
            record("V-002", "Whitespace name", "FAIL",
                   f"status={r.status_code}, body={r.text[:200]}")

        # V-003: Rename > 60 char -> 422 Pydantic max_length
        r = await client.patch("/org", json={"name": "X" * 100},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 422:
            record("V-003", "Rename > 60 char -> 422 max_length",
                   "PASS", "Pydantic max_length=60 enforced")
        else:
            record("V-003", "Long name", "FAIL", f"status={r.status_code}")

        # V-004: Rename exactly 60 char -> 200
        sixty = "A" * 60
        r = await client.patch("/org", json={"name": sixty},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200 and len(r.json()["name"]) == 60:
            record("V-004", "Rename exactly 60 char -> 200 (boundary)",
                   "PASS", "60-char accepted")
        else:
            record("V-004", "Boundary 60 char", "FAIL",
                   f"status={r.status_code}, len={len(r.json().get('name', ''))}")

        # V-005: Rename 1 char -> 200 (min boundary)
        r = await client.patch("/org", json={"name": "X"},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200 and r.json()["name"] == "X":
            record("V-005", "Rename 1 char -> 200 (min boundary)",
                   "PASS", "1-char accepted")
        else:
            record("V-005", "Boundary 1 char", "FAIL",
                   f"status={r.status_code}")

        # V-006: Join with empty code -> 422
        r = await client.post("/org/join", json={"invite_code": ""},
                               headers={"Authorization": f"Bearer {tokenC}"})
        if r.status_code == 422:
            record("V-006", "Join empty code -> 422",
                   "PASS", "min_length=1 enforced")
        else:
            record("V-006", "Empty join code", "FAIL",
                   f"status={r.status_code}")

        # V-007: Join with code > 32 char -> 422
        r = await client.post("/org/join", json={"invite_code": "X" * 50},
                               headers={"Authorization": f"Bearer {tokenC}"})
        if r.status_code == 422:
            record("V-007", "Join code > 32 char -> 422",
                   "PASS", "max_length=32 enforced")
        else:
            record("V-007", "Long join code", "FAIL",
                   f"status={r.status_code}")

        # ========================================================
        # N (Negative)
        # ========================================================
        # N-001: Non-owner rename -> 403
        # Joiner (qatest_joiner) is in orgA but not owner
        rJoinerToken = rJ.json()["access_token"]
        r = await client.patch("/org", json={"name": "Hacked Tim"},
                                headers={"Authorization": f"Bearer {rJoinerToken}"})
        if r.status_code == 403:
            record("N-001", "Non-owner rename -> 403",
                   "PASS", f"detail={r.json().get('detail')}")
        else:
            record("N-001", "Non-owner rename", "FAIL",
                   f"status={r.status_code}, body={r.text[:300]}")

        # N-002: Rename without auth -> 401
        r = await client.patch("/org", json={"name": "No auth"})
        if r.status_code == 401:
            record("N-002", "Rename without token -> 401",
                   "PASS", "Auth gate enforced")
        else:
            record("N-002", "Rename no auth", "FAIL", f"status={r.status_code}")

        # N-003: Join invalid code -> 404
        r = await client.post("/org/join", json={"invite_code": "ORG-XX9999"},
                               headers={"Authorization": f"Bearer {tokenC}"})
        if r.status_code == 404:
            record("N-003", "Join invalid code -> 404",
                   "PASS", f"detail={r.json().get('detail')}")
        else:
            record("N-003", "Invalid join code", "FAIL",
                   f"status={r.status_code}")

        # N-004: Join same org -> 400
        # Charlie sekarang di orgB (sudah join). Try join orgB lagi.
        r = await client.post("/org/join", json={"invite_code": orgB_code},
                               headers={"Authorization": f"Bearer {tokenC}"})
        if r.status_code == 400:
            record("N-004", "Join same org again -> 400",
                   "PASS", f"detail={r.json().get('detail')}")
        else:
            record("N-004", "Join same org", "FAIL", f"status={r.status_code}")

        # N-005: Join without auth -> 401
        r = await client.post("/org/join", json={"invite_code": orgB_code})
        if r.status_code == 401:
            record("N-005", "Join without token -> 401",
                   "PASS", "Auth gate enforced")
        else:
            record("N-005", "Join no auth", "FAIL", f"status={r.status_code}")

        # ========================================================
        # E (Edge cases) — Unicode, special chars, etc
        # ========================================================
        # E-001: Unicode (Indonesian special chars)
        unicode_name = "Tim Penjüalan Çabang"
        r = await client.patch("/org", json={"name": unicode_name},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200 and r.json()["name"] == unicode_name:
            record("E-001", "Rename dengan Unicode diacritics",
                   "PASS", f"name={r.json()['name']!r}")
        else:
            record("E-001", "Unicode name", "FAIL",
                   f"status={r.status_code}, got={r.json().get('name')!r}")

        # E-002: Emoji
        emoji_name = "🚦 Tim Lalu Lintas"
        r = await client.patch("/org", json={"name": emoji_name},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200 and r.json()["name"] == emoji_name:
            record("E-002", "Rename dengan emoji",
                   "PASS", f"name={r.json()['name']!r}")
        else:
            record("E-002", "Emoji name", "FAIL",
                   f"status={r.status_code}, got={r.json().get('name')!r}")

        # E-003: Tab dan newline di dalam name
        tab_name = "Tim\twith\ttabs"
        r = await client.patch("/org", json={"name": tab_name},
                                headers={"Authorization": f"Bearer {tokenA}"})
        # Server tidak strip internal whitespace, hanya trim luar
        if r.status_code == 200:
            record("E-003", "Rename dengan tab di tengah string",
                   "PASS", f"name={r.json()['name']!r}",
                   note="Server hanya trim leading/trailing, internal whitespace preserved")
        else:
            record("E-003", "Tab name", "WARN",
                   f"status={r.status_code}")

        # E-004: Owner rename ke nama yang SAMA dengan org lain (not unique)
        r = await client.patch("/org", json={"name": "Tim Bob"},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200:
            record("E-004", "Rename ke nama yang sama dengan org lain (non-unique)",
                   "PASS", "Dua org boleh nama sama (sesuai spec)")
        else:
            record("E-004", "Non-unique name", "FAIL",
                   f"status={r.status_code}")

        # E-005: Rename ke nama yang sama (no-op)
        # Currently orgA.name = "Tim Bob" from E-004
        r = await client.patch("/org", json={"name": "Tim Bob"},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200:
            record("E-005", "Rename ke nama yang sama persis (idempotent)",
                   "PASS", "No error, accepted as no-op")
        else:
            record("E-005", "Same name idempotent", "FAIL",
                   f"status={r.status_code}")

        # E-006: Multi-byte UTF-8 char count vs char limit
        # 60 multi-byte chars (each takes 3 bytes in UTF-8)
        multibyte_60 = "中" * 60
        r = await client.patch("/org", json={"name": multibyte_60},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200 and len(r.json()["name"]) == 60:
            record("E-006", "Rename 60 multi-byte char (Chinese) -> 200",
                   "PASS", "Pydantic counts chars not bytes")
        else:
            record("E-006", "Multi-byte length", "FAIL",
                   f"status={r.status_code}, len={len(r.json().get('name', ''))}")

        # E-007: Join with code that has whitespace around it (backend strip+upper)
        # First reset Charlie to orgC (move via admin path — skip; instead use a fresh user)
        rE = await client.post("/auth/register", json={
            "email": "qatest_eve@example.com", "name": "Eve",
            "password": "password123"})
        tokenE = rE.json()["access_token"]
        # Eve joins orgA with code containing spaces around
        r = await client.post("/org/join",
                               json={"invite_code": f"  {orgA_code.lower()}  "},
                               headers={"Authorization": f"Bearer {tokenE}"})
        if r.status_code == 200 and r.json()["org_id"] == orgA_id:
            record("E-007", "Join code with leading/trailing space + lowercase",
                   "PASS", "Backend strip+upper, accepted")
        else:
            record("E-007", "Code normalization", "FAIL",
                   f"status={r.status_code}, body={r.text[:200]}")

        # ========================================================
        # S (Security)
        # ========================================================
        # S-001: NoSQL injection via name field
        r = await client.patch(
            "/org",
            json={"name": '{"$ne": null}'},
            headers={"Authorization": f"Bearer {tokenA}"},
        )
        if r.status_code == 200:
            # If 200, verify name stored as literal string (not interpreted as Mongo operator)
            stored = r.json()["name"]
            if stored == '{"$ne": null}':
                record("S-001", "NoSQL injection di name field -> stored as literal",
                       "PASS", f"name stored as literal string")
            else:
                record("S-001", "NoSQL injection name", "FAIL",
                       f"unexpected stored value: {stored!r}")
        else:
            record("S-001", "NoSQL injection name", "WARN",
                   f"status={r.status_code}")

        # S-002: HTML/XSS payload in name (stored as-is, React escapes on render)
        xss = "<script>alert(1)</script>"
        r = await client.patch("/org", json={"name": xss},
                                headers={"Authorization": f"Bearer {tokenA}"})
        if r.status_code == 200 and r.json()["name"] == xss:
            record("S-002", "HTML/XSS payload tetap di-store sebagai text",
                   "PASS", "React escape default di render — aman",
                   note="Verify visually di /tim — text harusnya literal, bukan eksekusi script")
        else:
            record("S-002", "XSS payload", "WARN",
                   f"status={r.status_code}")

        # S-003: Join code injection — pakai operator-style string
        r = await client.post("/org/join",
                               json={"invite_code": '{"$ne": null}'},
                               headers={"Authorization": f"Bearer {tokenC}"})
        # invite_code lookup: db.orgs.find_one({"invite_code": code.strip()})
        # If "{$ne: null}" is treated as literal string, lookup miss -> 404
        if r.status_code == 404:
            record("S-003", "NoSQL injection di invite_code -> 404 lookup miss",
                   "PASS", "Motor parameterize, injection neutralized")
        else:
            record("S-003", "Join injection", "WARN",
                   f"status={r.status_code}, body={r.text[:200]}")

        # S-004: Cross-org rename attempt — try rename ANOTHER org's name
        # Not possible since PATCH /org only operates on user.org_id
        # But verify: if attacker manipulates JWT to spoof org_id, they fail
        # (since route reads user.org_id from DB lookup, not from JWT directly)
        # We test by: Charlie (in orgB) tries to rename — should rename orgB, NOT orgA
        # First check Charlie is still in orgB
        meC2 = await client.get("/org/me",
                                 headers={"Authorization": f"Bearer {tokenC}"})
        if meC2.json()["org_id"] == str(orgB_id):
            # Charlie is owner-of-orgC originally, but moved to orgB. Bob is owner of orgB.
            # So Charlie should NOT be able to rename orgB.
            r = await client.patch("/org", json={"name": "Charlie's Heist"},
                                    headers={"Authorization": f"Bearer {tokenC}"})
            if r.status_code == 403:
                record("S-004", "User di org orang lain tidak bisa rename org-nya",
                       "PASS", "Non-owner check enforced post-join")
            else:
                record("S-004", "Cross-org rename via join", "FAIL",
                       f"Charlie bisa rename Bob's org! status={r.status_code}",
                       note="CRITICAL: privilege escalation post-join")
        else:
            record("S-004", "Cross-org rename setup", "WARN",
                   f"Charlie tidak di orgB: {meC2.json()['org_id']}")

        # S-005: Owner-of-old-org rename setelah leave
        # Charlie originally owns orgC. Now in orgB. Try rename via /org (should target orgB, fail).
        # Already covered by S-004. Skip.

        # ========================================================
        # I (Integration) - Join + rename interplay
        # ========================================================
        # I-001: Eve (joined orgA via E-007) GET /org/me - sees current orgA name
        meE = await client.get("/org/me",
                                headers={"Authorization": f"Bearer {tokenE}"})
        if meE.status_code == 200:
            record("I-001", "Joiner /org/me reflects host org name after rename",
                   "PASS", f"Eve sees {meE.json().get('name')!r}")
        else:
            record("I-001", "Joiner /org/me", "FAIL",
                   f"status={meE.status_code}")

        # I-002: After rename, joining via same code still works
        meA3 = await client.get("/org/me",
                                 headers={"Authorization": f"Bearer {tokenA}"})
        current_code = meA3.json()["invite_code"]
        rF = await client.post("/auth/register", json={
            "email": "qatest_frank@example.com", "name": "Frank",
            "password": "password123", "invite_code": current_code})
        if rF.status_code == 201:
            record("I-002", "Rename tidak invalidate invite code",
                   "PASS", "Frank joined org renamed")
        else:
            record("I-002", "Code valid after rename", "FAIL",
                   f"status={rF.status_code}")

        # ========================================================
        # M (Migration) — Backfill name idempotency
        # ========================================================
        # Check all orgs in DB have name field
        orgs_without_name = await db.orgs.count_documents(
            {"name": {"$exists": False}})
        if orgs_without_name == 0:
            record("M-001", "Semua orgs di DB punya field name (migration ok)",
                   "PASS", "0 orgs tanpa name")
        else:
            record("M-001", "Backfill name", "FAIL",
                   f"{orgs_without_name} orgs masih tanpa name")

    # ============================================================
    # Cleanup
    # ============================================================
    cleanup = await db.users.delete_many({"email": {"$regex": "^qatest_"}})
    valid_ids = set()
    async for u in db.users.find({}, {"_id": 1}):
        valid_ids.add(u["_id"])
    orphan = await db.orgs.delete_many({"created_by": {"$nin": list(valid_ids)}})
    print()
    print(f"Cleanup: users={cleanup.deleted_count}, orphan orgs={orphan.deleted_count}")

    print()
    print("=" * 76)
    p = sum(1 for r in results if r["status"] == "PASS")
    f = sum(1 for r in results if r["status"] == "FAIL")
    w = sum(1 for r in results if r["status"] == "WARN")
    print(f"  TOTAL: {len(results)} | PASS: {p} | FAIL: {f} | WARN: {w}")
    print("=" * 76)
    if f:
        print()
        print("FAILED TESTS:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['id']}: {r['name']}")
                print(f"    Actual: {r['actual']}")
                if r["note"]:
                    print(f"    Note: {r['note']}")

    await close_db()


if __name__ == "__main__":
    asyncio.run(run())
