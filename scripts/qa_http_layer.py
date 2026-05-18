"""
QA HTTP layer tests via httpx.AsyncClient + ASGITransport.
Verify endpoints behave correctly with auth, validation, and permissions.

Pakai AsyncClient supaya share asyncio loop dengan Motor (TestClient sync
bikin loop berbeda yang konflik dengan AsyncIOMotorClient).
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
    results.append({"id": tc_id, "name": name, "status": status, "actual": actual})
    icon = "[PASS]" if status == "PASS" else ("[FAIL]" if status == "FAIL" else "[WARN]")
    print(f"{icon} {tc_id}: {name} -> {actual}")
    if note:
        print(f"      NOTE: {note}")


async def run():
    from httpx import AsyncClient, ASGITransport
    from services.database import connect_db, get_db, close_db
    from bson import ObjectId
    from fastapi import FastAPI

    await connect_db()
    db = get_db()
    assert db is not None
    await db.users.delete_many({"email": {"$regex": "^qahttp_"}})

    # Build minimal app — skip detector preload from main.py
    from routes.auth import router as auth_router
    from routes.admin import router as admin_router
    from routes.org import router as org_router
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(org_router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # ============================================================
        # H-001: Register tanpa invite_code -> 201
        # ============================================================
        r1 = await client.post("/auth/register", json={
            "email": "qahttp_admin@example.com",
            "name": "QA Admin",
            "password": "password123",
        })
        if r1.status_code == 201 and "access_token" in r1.json():
            record("H-001", "POST /auth/register tanpa invite_code -> 201", "PASS",
                   f"status=201, role={r1.json()['user']['role']}")
            # Promote test user ke admin role via DB direct (test setup).
            # Sistem sudah ada admin existing, jadi register normal dapat operator.
            await db.users.update_one(
                {"email": "qahttp_admin@example.com"},
                {"$set": {"role": "admin"}},
            )
            # Re-login untuk dapat token dengan role=admin (token carry role claim)
            r1b = await client.post("/auth/login", json={
                "email": "qahttp_admin@example.com",
                "password": "password123",
            })
            admin_token = r1b.json()["access_token"]
        else:
            record("H-001", "POST /auth/register tanpa invite_code", "FAIL",
                   f"status={r1.status_code}, body={r1.text[:200]}")
            admin_token = None

        admin_org_code = None
        admin_org_id = None
        if admin_token:
            r_me = await client.get("/org/me",
                                     headers={"Authorization": f"Bearer {admin_token}"})
            if r_me.status_code == 200:
                data = r_me.json()
                admin_org_code = data.get("invite_code")
                admin_org_id = data.get("org_id")
                record("H-002", "GET /org/me -> 200 dengan invite_code", "PASS",
                       f"code={admin_org_code}, is_owner={data.get('is_owner')}, members={data.get('member_count')}")
            else:
                record("H-002", "GET /org/me", "FAIL", f"status={r_me.status_code}")

        # H-003: Register dengan invalid invite_code -> 400
        r2 = await client.post("/auth/register", json={
            "email": "qahttp_invalid@example.com",
            "name": "Invalid",
            "password": "password123",
            "invite_code": "ORG-FAKE99",
        })
        if r2.status_code == 400 and "Kode undangan" in r2.text:
            record("H-003", "Register dengan invalid invite_code -> 400", "PASS",
                   f"detail={r2.json().get('detail')}")
        else:
            record("H-003", "Register invalid code", "FAIL",
                   f"status={r2.status_code}, body={r2.text[:200]}")

        # H-004: Register dengan valid code -> join admin org
        op_token = None
        if admin_org_code and admin_org_id:
            r3 = await client.post("/auth/register", json={
                "email": "qahttp_operator@example.com",
                "name": "QA Operator",
                "password": "password123",
                "invite_code": admin_org_code,
            })
            if r3.status_code == 201:
                op_doc = await db.users.find_one({"email": "qahttp_operator@example.com"})
                if op_doc and str(op_doc["org_id"]) == admin_org_id:
                    record("H-004", "Register dengan valid code -> join admin org", "PASS",
                           f"operator.org_id == admin.org_id = {admin_org_id}")
                    op_token = r3.json()["access_token"]
                else:
                    record("H-004", "Register valid code", "FAIL",
                           f"op.org_id={op_doc.get('org_id') if op_doc else None}")
            else:
                record("H-004", "Register valid code", "FAIL",
                       f"status={r3.status_code}, body={r3.text[:200]}")

        # H-005: Register dengan lowercase code (route handler upper)
        if admin_org_code:
            r4 = await client.post("/auth/register", json={
                "email": "qahttp_lower@example.com",
                "name": "Lower",
                "password": "password123",
                "invite_code": admin_org_code.lower(),
            })
            if r4.status_code == 201:
                record("H-005", "Register dengan lowercase code -> upper otomatis",
                       "PASS", "Route handler upper() sebelum service")
            else:
                record("H-005", "Lowercase code", "FAIL",
                       f"status={r4.status_code}, body={r4.text[:200]}")

        # H-006: Whitespace-only code -> bikin org baru
        r5 = await client.post("/auth/register", json={
            "email": "qahttp_ws@example.com",
            "name": "WS",
            "password": "password123",
            "invite_code": "   ",
        })
        if r5.status_code == 201:
            ws_doc = await db.users.find_one({"email": "qahttp_ws@example.com"})
            if ws_doc and str(ws_doc["org_id"]) != admin_org_id:
                record("H-006", "Register whitespace-only code -> org baru",
                       "PASS", f"new org={ws_doc['org_id']}")
            else:
                record("H-006", "Whitespace code", "FAIL",
                       "User join admin org (seharusnya bikin org baru)")
        else:
            record("H-006", "Whitespace code", "FAIL",
                   f"status={r5.status_code}, body={r5.text[:200]}")

        # H-007: Password < 6 char -> 422
        r6 = await client.post("/auth/register", json={
            "email": "qahttp_short@example.com",
            "name": "Short", "password": "1234",
        })
        if r6.status_code == 422:
            record("H-007", "Password < 6 char -> 422", "PASS", "status=422")
        else:
            record("H-007", "Password validation", "FAIL", f"status={r6.status_code}")

        # H-008: Duplicate email -> 409
        r7 = await client.post("/auth/register", json={
            "email": "qahttp_admin@example.com",
            "name": "Dup", "password": "password123",
        })
        if r7.status_code == 409:
            record("H-008", "Duplicate email -> 409", "PASS", "status=409")
        else:
            record("H-008", "Duplicate email", "FAIL", f"status={r7.status_code}")

        # H-009: invite_code > 32 char -> 422
        r8 = await client.post("/auth/register", json={
            "email": "qahttp_long@example.com",
            "name": "Long",
            "password": "password123",
            "invite_code": "X" * 100,
        })
        if r8.status_code == 422:
            record("H-009", "invite_code > 32 char -> 422 validation", "PASS",
                   "Pydantic max_length=32 enforced")
        else:
            record("H-009", "invite_code length", "FAIL",
                   f"status={r8.status_code}, body={r8.text[:200]}")

        # H-010: /org/me tanpa token -> 401
        r9 = await client.get("/org/me")
        if r9.status_code == 401:
            record("H-010", "GET /org/me tanpa token -> 401", "PASS", "Auth gate enforced")
        else:
            record("H-010", "Org/me tanpa auth", "FAIL", f"status={r9.status_code}")

        # H-011: Rotate tanpa token -> 401
        r10 = await client.post("/org/rotate-code")
        if r10.status_code == 401:
            record("H-011", "POST /org/rotate-code tanpa token -> 401", "PASS",
                   "Auth gate enforced")
        else:
            record("H-011", "Rotate tanpa auth", "FAIL", f"status={r10.status_code}")

        # H-012: Operator (non-owner) rotate -> 403
        if op_token:
            r11 = await client.post("/org/rotate-code",
                                     headers={"Authorization": f"Bearer {op_token}"})
            if r11.status_code == 403:
                record("H-012", "Operator non-owner rotate -> 403", "PASS",
                       "Permission check enforced")
            else:
                record("H-012", "Operator non-owner rotate", "FAIL",
                       f"status={r11.status_code}, body={r11.text[:300]}",
                       note="Operator yang JOIN org via code seharusnya tidak boleh rotate")

        # H-013: Admin (owner) rotate -> 200
        if admin_token:
            r12 = await client.post("/org/rotate-code",
                                     headers={"Authorization": f"Bearer {admin_token}"})
            if r12.status_code == 200 and r12.json().get("invite_code", "").startswith("ORG-"):
                new_code = r12.json()["invite_code"]
                record("H-013", "Admin (owner) rotate -> 200", "PASS", f"new={new_code}")
                admin_org_code = new_code
            else:
                record("H-013", "Admin rotate", "FAIL",
                       f"status={r12.status_code}, body={r12.text[:200]}")

        # H-014: GET /org/list as admin -> 200 list
        if admin_token:
            r13 = await client.get("/org/list",
                                    headers={"Authorization": f"Bearer {admin_token}"})
            if r13.status_code == 200 and isinstance(r13.json(), list):
                record("H-014", "GET /org/list as admin -> 200 list", "PASS",
                       f"count={len(r13.json())}")
            else:
                record("H-014", "Org list as admin", "FAIL",
                       f"status={r13.status_code}, body={r13.text[:200]}")

        # H-015: GET /org/list as operator -> 403
        if op_token:
            r14 = await client.get("/org/list",
                                    headers={"Authorization": f"Bearer {op_token}"})
            if r14.status_code == 403:
                record("H-015", "GET /org/list as operator -> 403", "PASS",
                       "Admin-only enforced")
            else:
                record("H-015", "Org list as operator", "FAIL",
                       f"status={r14.status_code}")

        # H-016: Admin assign user ke org lain
        if admin_token and op_token:
            op_doc = await db.users.find_one({"email": "qahttp_operator@example.com"})
            ws_doc = await db.users.find_one({"email": "qahttp_ws@example.com"})
            if op_doc and ws_doc:
                r15 = await client.patch(
                    f"/admin/users/{op_doc['_id']}/org",
                    json={"org_id": str(ws_doc["org_id"])},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                if r15.status_code == 200:
                    moved = await db.users.find_one({"_id": op_doc["_id"]})
                    if moved["org_id"] == ws_doc["org_id"]:
                        record("H-016", "Admin assign user ke org lain -> 200", "PASS",
                               f"operator moved")
                    else:
                        record("H-016", "Admin assign", "FAIL", "DB tidak update")
                else:
                    record("H-016", "Admin assign", "FAIL",
                           f"status={r15.status_code}, body={r15.text[:300]}")

        # H-017: Admin assign dengan invalid org_id format -> 400
        if admin_token:
            op_doc = await db.users.find_one({"email": "qahttp_operator@example.com"})
            if op_doc:
                r16 = await client.patch(
                    f"/admin/users/{op_doc['_id']}/org",
                    json={"org_id": "not-an-objectid"},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                if r16.status_code == 400:
                    record("H-017", "org_id format invalid -> 400", "PASS",
                           f"detail={r16.json().get('detail')}")
                else:
                    record("H-017", "Invalid org_id format", "FAIL",
                           f"status={r16.status_code}, body={r16.text[:200]}")

        # H-018: Admin assign ke org non-existent -> 404
        if admin_token:
            op_doc = await db.users.find_one({"email": "qahttp_operator@example.com"})
            if op_doc:
                r17 = await client.patch(
                    f"/admin/users/{op_doc['_id']}/org",
                    json={"org_id": str(ObjectId())},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                if r17.status_code == 404:
                    record("H-018", "Admin assign ke org non-existent -> 404", "PASS",
                           "Non-existent org rejected")
                else:
                    record("H-018", "Non-existent org", "FAIL",
                           f"status={r17.status_code}, body={r17.text[:200]}")

        # H-019: Admin assign self -> 400
        if admin_token:
            admin_doc = await db.users.find_one({"email": "qahttp_admin@example.com"})
            if admin_doc:
                r18 = await client.patch(
                    f"/admin/users/{admin_doc['_id']}/org",
                    json={"org_id": str(admin_doc["org_id"])},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                if r18.status_code == 400 and "diri sendiri" in r18.text:
                    record("H-019", "Admin assign self -> 400", "PASS",
                           "Self-action blocked")
                else:
                    record("H-019", "Admin self-assign", "FAIL",
                           f"status={r18.status_code}, body={r18.text[:200]}")

        # H-020: Operator akses admin endpoint -> 403
        if op_token:
            admin_doc = await db.users.find_one({"email": "qahttp_admin@example.com"})
            if admin_doc:
                r19 = await client.patch(
                    f"/admin/users/{admin_doc['_id']}/org",
                    json={"org_id": str(ObjectId())},
                    headers={"Authorization": f"Bearer {op_token}"},
                )
                if r19.status_code == 403:
                    record("H-020", "Operator akses admin endpoint -> 403", "PASS",
                           "require_admin enforced")
                else:
                    record("H-020", "Operator unauthorized", "FAIL",
                           f"status={r19.status_code}")

        # H-021: Admin assign ke same org -> 200 idempotent
        if admin_token:
            op_doc = await db.users.find_one({"email": "qahttp_operator@example.com"})
            if op_doc:
                r20 = await client.patch(
                    f"/admin/users/{op_doc['_id']}/org",
                    json={"org_id": str(op_doc["org_id"])},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                if r20.status_code == 200:
                    record("H-021", "Admin assign ke same org -> 200 idempotent", "PASS",
                           "No-op handled")
                else:
                    record("H-021", "Same-org no-op", "FAIL",
                           f"status={r20.status_code}, body={r20.text[:200]}")

        # H-022: SQL/NoSQL injection attempt via invite_code
        r_inj = await client.post("/auth/register", json={
            "email": "qahttp_inj@example.com",
            "name": "Inject",
            "password": "password123",
            "invite_code": '{"$ne": null}',  # Mongo operator injection attempt
        })
        if r_inj.status_code == 400:
            record("H-022", "NoSQL injection via invite_code -> 400", "PASS",
                   "Operator-style string treated as literal, lookup miss")
        elif r_inj.status_code == 201:
            inj_doc = await db.users.find_one({"email": "qahttp_inj@example.com"})
            if inj_doc and str(inj_doc["org_id"]) != admin_org_id:
                record("H-022", "NoSQL injection -> safe (new org)", "PASS",
                       "Injection neutralized (validation+typing)")
            else:
                record("H-022", "NoSQL injection", "FAIL",
                       "Injection joined admin org!")
        elif r_inj.status_code == 422:
            record("H-022", "NoSQL injection -> 422 validation", "PASS",
                   "Pydantic blocked malformed input")
        else:
            record("H-022", "NoSQL injection", "WARN",
                   f"status={r_inj.status_code}, body={r_inj.text[:200]}")

    # Cleanup
    cleanup_users = await db.users.delete_many({"email": {"$regex": "^qahttp_"}})
    # Find orgs whose created_by no longer exists
    valid_user_ids = set()
    async for u in db.users.find({}, {"_id": 1}):
        valid_user_ids.add(u["_id"])
    orphan = await db.orgs.delete_many({"created_by": {"$nin": list(valid_user_ids)}})
    print()
    print(f"Cleanup: users={cleanup_users.deleted_count}, orphan orgs={orphan.deleted_count}")

    # Summary
    print()
    print("=" * 72)
    p = sum(1 for r in results if r["status"] == "PASS")
    f = sum(1 for r in results if r["status"] == "FAIL")
    w = sum(1 for r in results if r["status"] == "WARN")
    print(f"  TOTAL: {len(results)} | PASS: {p} | FAIL: {f} | WARN: {w}")
    print("=" * 72)
    if f:
        print()
        print("FAILED TESTS:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['id']}: {r['name']} -> {r['actual']}")

    await close_db()


if __name__ == "__main__":
    asyncio.run(run())
