@echo off
:: ============================================================
:: start-backend.bat — Menjalankan Backend FastAPI di Laptop
:: ============================================================
::
:: DESKRIPSI:
::   Script ini mengaktifkan virtual environment Python lalu
::   menjalankan server FastAPI di port 3219.
::   Jalankan script ini SEBELUM expose-backend.bat.
::
:: PRASYARAT:
::   - Python 3.10+ sudah terinstall
::   - Virtual environment sudah dibuat di folder ini
::     (folder "venv" atau ".venv" atau "env")
::   - Semua dependency sudah diinstall:
::       pip install -r requirements.txt
::   - File .env sudah diisi (MONGO_URI, CORS_ORIGINS, dll.)
::   - ffmpeg sudah terinstall:
::       winget install ffmpeg
::       atau: choco install ffmpeg
::
:: CARA MENJALANKAN:
::   Double-click file ini, atau jalankan dari terminal:
::     api-traffic-counter\start-backend.bat
:: ============================================================

title Traffic Counter Backend

echo.
echo ============================================================
echo   Traffic Counter API - Backend Server
echo ============================================================
echo.

:: Pindah ke folder tempat script ini berada (api-traffic-counter/)
cd /d "%~dp0"

:: ── Aktifkan virtual environment ─────────────────────────────
:: Coba berbagai lokasi venv yang umum
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan virtual environment: venv\
    call venv\Scripts\activate.bat
    goto :start_server
)
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan virtual environment: .venv\
    call .venv\Scripts\activate.bat
    goto :start_server
)
if exist "env\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan virtual environment: env\
    call env\Scripts\activate.bat
    goto :start_server
)

echo [WARNING] Virtual environment tidak ditemukan di folder:
echo           venv\, .venv\, atau env\
echo           Melanjutkan dengan Python global...
echo.

:start_server
:: ── Validasi file .env ───────────────────────────────────────
if not exist ".env" (
    echo [ERROR] File .env tidak ditemukan!
    echo         Salin .env.example menjadi .env lalu isi nilainya.
    echo.
    pause
    exit /b 1
)

:: ── Jalankan server ──────────────────────────────────────────
echo [INFO] Memulai server di http://0.0.0.0:3219
echo [INFO] API Docs tersedia di http://localhost:3219/docs
echo [INFO] Tekan Ctrl+C untuk menghentikan server
echo.
echo ============================================================
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 3219 --reload

:: Jika server berhenti karena error, tampilkan pesan
echo.
echo ============================================================
echo [INFO] Server telah berhenti.
pause
