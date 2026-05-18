"""
QA test script untuk fitur multi-tenant + invite code + admin assign.
Idempotent — bersihkan dirinya sendiri. Dijalankan oleh QA Engineer
untuk validasi end-to-end sebelum deploy ke production.

Run: uv run python scripts/qa_multitenant.py
"""
from __future__ import annotations
import asyncio
import sys
import re
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
    results.append({"id": tc_id, "name": name, "status": status, "actual": actual, "note": note})
    icon = "[PASS]" if status == "PASS" else ("[FAIL]" if status == "FAIL" else "[WARN]")
    print(f"{icon} {tc_id}: {name} -> {actual}")
    if note:
        print(f"      NOTE: {note}")


async def run():
    from services.database import connect_db, get_db, close_db
    from services.org_service import (
        create_org_for_user, find_org_by_invite_code,
        rotate_invite_code, is_org_owner, generate_invite_code,
        _INVITE_CODE_MAX_RETRY,
    )
    from services.auth_service import create_user, InviteCodeNotFound
    from dependencies.auth import scope_filter
    from bson import ObjectId

    await connect_db()
    db = get_db()
    assert db is not None

    await db.users.delete_many({"email": {"$regex": "^qatest_"}})
    await db.branches.delete_many({"name": {"$regex": "^QATEST-"}})

    # ============================================================
    # F (Functional) — happy path
    # ============================================================
    try:
        code = generate_invite_code()
        if re.match(r"^ORG-[A-Z0-9]{6}$", code):
            record("F-001", "Format generate_invite_code ORG-XXXXXX", "PASS", code)
        else:
            record("F-001", "Format generate_invite_code ORG-XXXXXX", "FAIL", code)
    except Exception as e:
        record("F-001", "Format generate_invite_code ORG-XXXXXX", "FAIL", str(e))

    solo_user_id = None
    solo_org_id = None
    solo_code = None
    try:
        u = await create_user("qatest_solo@example.com", "Solo User", "password123")
        org_doc = await db.orgs.find_one({"_id": u["org_id"]})
        if org_doc and org_doc["created_by"] == u["_id"] and org_doc["invite_code"].startswith("ORG-"):
            record("F-002", "Register tanpa invite_code -> bikin org baru", "PASS",
                   f"org={u['org_id']}, code={org_doc['invite_code']}")
        else:
            record("F-002", "Register tanpa invite_code", "FAIL", str(org_doc))
        solo_user_id = u["_id"]
        solo_org_id = u["org_id"]
        solo_code = org_doc["invite_code"]
    except Exception as e:
        record("F-002", "Register tanpa invite_code", "FAIL", str(e))
        await close_db()
        return

    try:
        u2 = await create_user("qatest_join@example.com", "Joiner", "password123",
                                invite_code=solo_code)
        if u2["org_id"] == solo_org_id:
            record("F-003", "Register dengan invite_code valid -> join org", "PASS",
                   f"joiner.org_id == owner.org_id == {solo_org_id}")
        else:
            record("F-003", "Register dengan invite_code valid", "FAIL",
                   f"joiner.org_id={u2['org_id']}, expected {solo_org_id}")
    except Exception as e:
        record("F-003", "Register dengan invite_code valid", "FAIL", str(e))

    try:
        new_code = await rotate_invite_code(solo_org_id)
        rotated = await db.orgs.find_one({"_id": solo_org_id})
        if rotated["invite_code"] == new_code and new_code != solo_code:
            record("F-004", "Rotate invite_code -> code baru", "PASS",
                   f"{solo_code} -> {new_code}")
            old_lookup = await find_org_by_invite_code(solo_code)
            if old_lookup is None:
                record("F-005", "Kode lama instant invalid setelah rotate", "PASS",
                       "find_org_by_invite_code(old) = None")
            else:
                record("F-005", "Kode lama invalid setelah rotate", "FAIL",
                       f"old_code masih valid: {old_lookup['_id']}")
            new_lookup = await find_org_by_invite_code(new_code)
            if new_lookup and new_lookup["_id"] == solo_org_id:
                record("F-006", "Kode baru valid setelah rotate", "PASS", str(new_lookup["_id"]))
            else:
                record("F-006", "Kode baru valid setelah rotate", "FAIL", str(new_lookup))
            solo_code = new_code
        else:
            record("F-004", "Rotate invite_code", "FAIL", str(rotated))
    except Exception as e:
        record("F-004", "Rotate invite_code", "FAIL", str(e))

    # ============================================================
    # N (Negative) — invalid input
    # ============================================================
    try:
        await create_user("qatest_invalid@example.com", "Invalid", "password123",
                          invite_code="ORG-NOTEXIST")
        record("N-001", "Register dengan invite_code tidak terdaftar -> reject",
               "FAIL", "Tidak ada exception")
    except InviteCodeNotFound as e:
        record("N-001", "Register dengan invite_code tidak terdaftar -> reject",
               "PASS", f"InviteCodeNotFound raised")
    except Exception as e:
        record("N-001", "Register dengan invite_code tidak terdaftar",
               "FAIL", f"Wrong exception: {type(e).__name__}: {e}")

    try:
        await create_user("qatest_lower@example.com", "Lower", "password123",
                          invite_code=solo_code.lower())
        record("N-002", "Service create_user dengan lowercase code", "WARN",
               "Service accept lowercase (case-insensitive)",
               note="Tapi route handler upper sebelum panggil, jadi HTTP flow aman.")
    except InviteCodeNotFound:
        record("N-002", "Service direct call dengan lowercase code -> reject",
               "PASS", "Service expect uppercase (case-sensitive lookup)",
               note="Frontend & route handler sudah .upper() sebelum panggil service. HTTP path aman.")
    except Exception as e:
        record("N-002", "Service direct call lowercase code", "WARN", str(e))

    try:
        await create_user("qatest_long@example.com", "Long", "password123",
                          invite_code="X" * 1000)
        record("N-003", "Register dengan code sangat panjang (1000 char)",
               "FAIL", "Diterima tanpa error")
    except InviteCodeNotFound:
        record("N-003", "Register dengan code sangat panjang -> reject (lookup miss)",
               "PASS", "InviteCodeNotFound (treated as not-found)")
    except Exception as e:
        record("N-003", "Register dengan code sangat panjang", "WARN", str(e))

    # ============================================================
    # V (Validation) — data integrity
    # ============================================================
    try:
        await db.orgs.insert_one({
            "_id": ObjectId(),
            "invite_code": solo_code,
            "created_at": "2026-05-15",
            "created_by": ObjectId(),
        })
        record("V-001", "Compound unique invite_code (global)", "FAIL",
               "Duplicate code diterima!")
    except Exception as e:
        if "duplicate" in str(e).lower() or "E11000" in str(e):
            record("V-001", "Compound unique invite_code (global)", "PASS",
                   "DuplicateKeyError raised")
        else:
            record("V-001", "Compound unique invite_code", "FAIL", str(e))

    test_branch_name = f"QATEST-Branch-{solo_org_id}"
    try:
        await db.branches.insert_one({
            "_id": ObjectId(), "org_id": solo_org_id,
            "name": test_branch_name, "address": "",
            "created_at": "2026-05-15", "created_by": "x",
            "updated_by": None, "updated_at": None,
        })
        try:
            await db.branches.insert_one({
                "_id": ObjectId(), "org_id": solo_org_id,
                "name": test_branch_name,
                "address": "", "created_at": "2026-05-15",
                "created_by": "y", "updated_by": None, "updated_at": None,
            })
            record("V-002", "Same name in same org -> reject", "FAIL",
                   "Duplicate accepted")
        except Exception as e:
            if "duplicate" in str(e).lower() or "E11000" in str(e):
                record("V-002", "Same name in same org -> reject (compound unique)",
                       "PASS", "DuplicateKeyError")
            else:
                record("V-002", "Same name in same org", "FAIL", str(e))

        other_org_id = ObjectId()
        await db.branches.insert_one({
            "_id": ObjectId(), "org_id": other_org_id,
            "name": test_branch_name, "address": "",
            "created_at": "2026-05-15", "created_by": "z",
            "updated_by": None, "updated_at": None,
        })
        record("V-003", "Same name in different orgs -> allow",
               "PASS", "Compound unique honors org_id")
        await db.branches.delete_many({"org_id": other_org_id})
    except Exception as e:
        record("V-002", "Branches uniqueness setup", "FAIL", str(e))

    # ============================================================
    # S (Security) — permission boundaries
    # ============================================================
    try:
        org_doc = await db.orgs.find_one({"_id": solo_org_id})
        owner_user = {"_id": solo_user_id, "role": "operator"}
        other_user = {"_id": ObjectId(), "role": "operator"}
        ok_owner = is_org_owner(owner_user, org_doc)
        ok_other = is_org_owner(other_user, org_doc)
        if ok_owner is True and ok_other is False:
            record("S-001", "is_org_owner permission", "PASS",
                   "True utk owner, False utk non-owner")
        else:
            record("S-001", "is_org_owner permission", "FAIL",
                   f"owner={ok_owner}, other={ok_other}")
    except Exception as e:
        record("S-001", "is_org_owner permission", "FAIL", str(e))

    try:
        admin_dict = {"role": "admin", "org_id": ObjectId()}
        op_dict = {"role": "operator", "org_id": solo_org_id}
        f_admin = scope_filter(admin_dict)
        f_op = scope_filter(op_dict)
        if f_admin == {} and f_op == {"org_id": solo_org_id}:
            record("S-002", "scope_filter (admin skip, operator scoped)", "PASS",
                   f"admin=dict empty, operator={f_op}")
        else:
            record("S-002", "scope_filter behaviour", "FAIL",
                   f"admin={f_admin}, op={f_op}")
    except Exception as e:
        record("S-002", "scope_filter behaviour", "FAIL", str(e))

    try:
        await db.branches.insert_one({
            "_id": ObjectId(), "org_id": solo_org_id,
            "name": f"QATEST-Iso-{solo_org_id}",
            "address": "", "created_at": "2026-05-15",
            "created_by": "x", "updated_by": None, "updated_at": None,
        })
        other_op = {"role": "operator", "org_id": ObjectId()}
        cnt_other = await db.branches.count_documents(
            {**scope_filter(other_op), "name": {"$regex": "^QATEST-Iso-"}})
        owner = {"role": "operator", "org_id": solo_org_id}
        cnt_owner = await db.branches.count_documents(
            {**scope_filter(owner), "name": {"$regex": "^QATEST-Iso-"}})
        if cnt_other == 0 and cnt_owner == 1:
            record("S-003", "Cross-org isolation utk branches via scope_filter",
                   "PASS", f"other_org_sees=0, owner_sees=1")
        else:
            record("S-003", "Cross-org isolation",
                   "FAIL", f"other={cnt_other}, owner={cnt_owner}")
    except Exception as e:
        record("S-003", "Cross-org isolation", "FAIL", str(e))

    # ============================================================
    # E (Edge case)
    # ============================================================
    if _INVITE_CODE_MAX_RETRY >= 3:
        record("E-001", "Collision retry constant >= 3", "PASS",
               f"_INVITE_CODE_MAX_RETRY={_INVITE_CODE_MAX_RETRY}")
    else:
        record("E-001", "Collision retry constant", "FAIL",
               f"_MAX_RETRY={_INVITE_CODE_MAX_RETRY} terlalu rendah")

    codes_generated = {generate_invite_code() for _ in range(100)}
    if len(codes_generated) == 100:
        record("E-002", "100 generate_invite_code -> 0 collision", "PASS",
               "Random distribution OK")
    else:
        record("E-002", "100 generate_invite_code collision check", "WARN",
               f"Only {len(codes_generated)} unique out of 100")

    try:
        await create_user("qatest_empty@example.com", "Empty", "password123",
                          invite_code="")
        empty_user = await db.users.find_one({"email": "qatest_empty@example.com"})
        if empty_user and empty_user.get("org_id") and empty_user["org_id"] != solo_org_id:
            record("E-003", "Empty string invite_code -> bikin org baru",
                   "PASS", f"new org={empty_user['org_id']}")
        else:
            record("E-003", "Empty string invite_code", "FAIL", str(empty_user))
    except Exception as e:
        record("E-003", "Empty string invite_code", "FAIL", str(e))

    # ============================================================
    # P (Performance/Index)
    # ============================================================
    orgs_idx = await db.orgs.index_information()
    if "invite_code_unique" in orgs_idx and "created_by_idx" in orgs_idx:
        record("P-001", "Indexes orgs (invite_code_unique, created_by_idx) ada",
               "PASS", str(sorted(orgs_idx.keys())))
    else:
        record("P-001", "Indexes orgs", "FAIL", str(orgs_idx))

    branches_idx = await db.branches.index_information()
    if "org_name_unique" in branches_idx:
        record("P-002", "Compound unique (org_id, name) di branches ada",
               "PASS", "org_name_unique present")
    else:
        record("P-002", "Compound unique branches", "FAIL", str(branches_idx))

    # ============================================================
    # Cleanup
    # ============================================================
    cleanup_users = await db.users.delete_many({"email": {"$regex": "^qatest_"}})
    cleanup_branches = await db.branches.delete_many({"name": {"$regex": "^QATEST-"}})
    cleanup_orgs = await db.orgs.delete_many({"_id": solo_org_id})
    # Also delete the new org that got created from E-003
    cleanup_orgs2 = await db.orgs.delete_many({"created_at": "2026-05-15"})
    print()
    print(f"Cleanup: users={cleanup_users.deleted_count}, "
          f"branches={cleanup_branches.deleted_count}, "
          f"orgs={cleanup_orgs.deleted_count + cleanup_orgs2.deleted_count}")

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
