# QA Scripts — Traffic Detection PSM

Folder ini berisi **QA script on-demand** untuk validasi end-to-end fitur backend
dan frontend. Berbeda dengan unit/integration test:

| | Folder ini (`qa/`) | Framework tests (`tests/`, `src/test/`) |
|---|---|---|
| Runner | Manual: `uv run python qa/<script>.py` | Auto: `uv run pytest`, `npm run test` |
| Butuh server hidup? | **Ya** — backend di `:3219` & MongoDB connected | Tidak (mock) |
| Reset DB sendiri? | Mostly yes (idempotent), pakai email/branch random | N/A — mock |
| Tujuan | Pre-deploy smoke + QA pack | Regression on commit |

Skrip di folder ini **tidak** ter-auto-discover oleh `pytest` (file pattern di
`pytest.ini` hanya match `test_*.py`).

---

## Inventory

| Skrip | Cakupan | Bahasa | Prasyarat |
|-------|---------|--------|-----------|
| [qa_auth.py](qa_auth.py) | Register, login, refresh, /auth/me — positive, validation, negative, security | Python (httpx) | Backend hidup + MongoDB |
| [qa_phase_a_b.py](qa_phase_a_b.py) | Branch CRUD + endpoint gating (admin vs operator) | Python (httpx + websockets) | Backend hidup + MongoDB |
| [qa_multitenant.py](qa_multitenant.py) | Multi-tenant scope, invite code, admin assign | Python (httpx + Motor langsung) | Backend hidup + MongoDB |
| [qa_http_layer.py](qa_http_layer.py) | Endpoint behavior via ASGITransport (in-process) | Python (httpx in-process) | MongoDB; backend tidak perlu nyala |
| [qa_join_rename.py](qa_join_rename.py) | Self-service join via invite code + rename org by owner/admin | Python (httpx + Motor) | Backend hidup + MongoDB |
| [e2e_auth.mjs](e2e_auth.mjs) | Browser real (Playwright): register + login flow, localStorage, redirect | Node.js (Playwright) | ⚠️ **lihat catatan khusus di bawah** |

---

## Cara Menjalankan

### Backend QA scripts (Python)

Semua script Python pakai pola `Path(__file__).parent.parent` untuk menemukan
root `traffic-detection-api/`, jadi sys.path tetap valid setelah pindahan ke
`qa/`. Jalankan **dari folder `traffic-detection-api/`**:

```bash
# Pastikan backend hidup di terminal lain dulu (kecuali qa_http_layer.py yang in-process)
uv run main.py    # terminal 1

# Lalu di terminal 2:
uv run python qa/qa_auth.py
uv run python qa/qa_phase_a_b.py
uv run python qa/qa_multitenant.py
uv run python qa/qa_join_rename.py
uv run python qa/qa_http_layer.py     # tidak butuh backend nyala

# Override URL backend bila berbeda:
QA_BASE=http://127.0.0.1:3219 uv run python qa/qa_auth.py
```

Tiap skrip mencetak `PASS` / `FAIL` per kasus dan keluar dengan **exit code =
jumlah test yang gagal** (0 = semua hijau), sehingga bisa dipakai di CI.

### Frontend E2E (`e2e_auth.mjs`)

⚠️ **Constraint:** Skrip ini import `playwright` dan `mongodb` dari
`frontend-traffic-counter/node_modules`. Node mencari `node_modules` mulai dari
**direktori skrip ke atas** (bukan CWD) — kalau file di `traffic-detection-api/qa/`,
Node tidak akan menemukan Playwright. Dua cara aman:

**Opsi 1 — `NODE_PATH` env (paling sederhana):**

PowerShell (Windows):
```powershell
$env:NODE_PATH = "d:\computer-vision\frontend-traffic-counter\node_modules"
node d:\computer-vision\traffic-detection-api\qa\e2e_auth.mjs
```

bash / zsh:
```bash
NODE_PATH=$(realpath ../frontend-traffic-counter/node_modules) \
  node traffic-detection-api/qa/e2e_auth.mjs
```

**Opsi 2 — Symlink (sekali setup, lalu jalankan biasa):**

```powershell
# PowerShell, sekali saja, dari root repo
New-Item -ItemType SymbolicLink `
  -Path traffic-detection-api\qa\node_modules `
  -Target ..\..\frontend-traffic-counter\node_modules
```

```bash
# POSIX, sekali saja
ln -s ../../frontend-traffic-counter/node_modules \
      traffic-detection-api/qa/node_modules
```

Setelah symlink dibuat, cukup: `node qa/e2e_auth.mjs` dari `traffic-detection-api/`.

**Prasyarat:** frontend dev server hidup di `:9321` (`npm run dev` dari
`frontend-traffic-counter/`) **dan** backend hidup di `:3219`.

```bash
# Env var opsional
$env:FRONTEND_URL = "http://localhost:9321"   # default
$env:MONGO_URI    = "mongodb://..."           # untuk cleanup user pasca-test
$env:MONGO_DB_NAME = "traffic_counter"        # default
```

---

## Konvensi

- **Nama file:** `qa_<feature>.py` atau `qa_<phase>.py`. Frontend E2E: `e2e_<feature>.mjs`.
- **Idempotent:** test data harus pakai suffix random (mis. `secrets.token_hex(4)`)
  + bersihkan diri sendiri di akhir. Jangan tinggalkan record permanen.
- **Exit code:** 0 = semua hijau; >0 = jumlah test yang gagal. Konsisten dengan CI.
- **Stdout:** baris `PASS` dan `FAIL` mudah di-grep. Hindari logging non-essential.
- **Bahasa output:** Indonesia untuk pesan user-facing; English untuk komentar kode.

## Cross-Reference

Test berbasis framework (auto-run) tetap di lokasi standar:

- **Backend pytest** — [`traffic-detection-api/tests/`](../tests/)
  - `test_auth_service.py`, `test_logging.py`, `test_video_jobs.py`
  - `conftest.py` (fixtures: `mock_db`, dll.)
- **Frontend vitest** — [`frontend-traffic-counter/src/test/`](../../frontend-traffic-counter/src/test/)
  - `example.test.ts`, `job-persistence.test.ts`, `bantuan.test.tsx`
  - `setup.ts` (matchMedia + localStorage polyfill)
