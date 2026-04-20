@echo off
:: ============================================================
:: expose-backend.bat — Expose Backend ke Internet (GRATIS)
::                      via Cloudflare Tunnel
:: ============================================================
::
:: DESKRIPSI:
::   Script ini membuat tunnel dari internet ke backend lokal
::   di port 3219 menggunakan Cloudflare Tunnel (cloudflared).
::   Setelah dijalankan, URL publik seperti:
::     https://xxxx-xxxx-xxxx.trycloudflare.com
::   akan muncul — URL inilah yang dipakai sebagai
::   BACKEND_PUBLIC_URL di traffic-pulse/deploy-frontend.sh
::   dan CORS_ORIGINS di .env backend.
::
:: PRASYARAT:
::   - cloudflared sudah terinstall.
::     Cara install (pilih salah satu):
::       winget install Cloudflare.cloudflared
::       atau: choco install cloudflared
::     Verifikasi: cloudflared --version
::   - start-backend.bat sudah dijalankan lebih dulu
::     dan backend sudah berjalan di port 3219
::
:: CARA MENJALANKAN:
::   Jalankan di terminal TERPISAH dari start-backend.bat:
::     api-traffic-counter\expose-backend.bat
::
:: CATATAN PENTING:
::   - URL berubah SETIAP KALI script ini dijalankan ulang.
::   - Jika URL berubah, Anda harus:
::     1. Update CORS_ORIGINS di api-traffic-counter/.env
::        lalu restart start-backend.bat
::     2. Jalankan ulang traffic-pulse/deploy-frontend.sh
::        dengan URL baru di variabel BACKEND_PUBLIC_URL
::   - Untuk URL yang TIDAK berubah (permanent), daftar akun
::     Cloudflare gratis dan buat Named Tunnel.
:: ============================================================

title Cloudflare Tunnel - Backend Expose

echo.
echo ============================================================
echo   Cloudflare Tunnel - Expose Backend ke Internet
echo ============================================================
echo.

:: ── Validasi cloudflared sudah terinstall ────────────────────
where cloudflared >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cloudflared tidak ditemukan!
    echo.
    echo         Install dengan salah satu perintah berikut:
    echo           winget install Cloudflare.cloudflared
    echo           atau: choco install cloudflared
    echo.
    echo         Setelah install, jalankan ulang script ini.
    pause
    exit /b 1
)

:: ── Validasi backend sudah berjalan ──────────────────────────
echo [INFO] Memeriksa apakah backend sudah berjalan di port 3219...
curl -s --max-time 3 http://localhost:3219/ >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Backend tidak terdeteksi di port 3219.
    echo           Pastikan start-backend.bat sudah dijalankan.
    echo           Melanjutkan membuka tunnel...
    echo.
) else (
    echo [OK] Backend berjalan di port 3219
    echo.
)

:: ── Instruksi ─────────────────────────────────────────────────
echo [INFO] Membuka tunnel ke http://localhost:3219 ...
echo.
echo ┌─────────────────────────────────────────────────────────┐
echo │  SETELAH URL MUNCUL:                                    │
echo │                                                         │
echo │  1. Salin URL (format: https://xxxx.trycloudflare.com)  │
echo │  2. Tempel ke traffic-pulse/deploy-frontend.sh:         │
echo │       BACKEND_PUBLIC_URL="https://xxxx.trycloudflare..."│
echo │  3. Tempel ke api-traffic-counter/.env:                 │
echo │       CORS_ORIGINS=http://IP_VPS_ANDA                   │
echo │     lalu restart start-backend.bat                      │
echo │  4. Jalankan: bash traffic-pulse/deploy-frontend.sh     │
echo │                                                         │
echo │  Jangan tutup jendela ini selama backend dipakai!       │
echo └─────────────────────────────────────────────────────────┘
echo.
echo ============================================================
echo.

cloudflared tunnel --url http://localhost:3219

echo.
echo ============================================================
echo [INFO] Tunnel telah ditutup.
pause
