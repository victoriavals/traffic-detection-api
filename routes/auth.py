"""
Authentication Routes — JWT-based register & login.

Endpoints:
- POST /auth/register  → Daftar akun baru, kembalikan JWT
- POST /auth/login     → Login, kembalikan JWT
- GET  /auth/me        → Info user dari token aktif
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from constant_var import (
    JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
    debug_info, debug_error,
)
from services.database import get_db

router = APIRouter(prefix="/auth", tags=["🔐 Auth"])

# ─── Crypto helpers ────────────────────────────────────────────────────────────
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def _create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload["exp"] = expire
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# ─── Schemas ───────────────────────────────────────────────────────────────────

class RegisterBody(BaseModel):
    name: str
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


# ─── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Daftar akun baru",
)
async def register(body: RegisterBody) -> TokenResponse:
    """Buat akun baru dan kembalikan JWT token."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    # Validasi input
    name = body.name.strip()
    email = body.email.lower().strip()

    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Nama minimal 2 karakter")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Format email tidak valid")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password minimal 6 karakter")

    # Cek duplikasi email
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=409, detail="Email sudah terdaftar")

    now = datetime.now(timezone.utc)
    result = await db.users.insert_one({
        "name": name,
        "email": email,
        "password": _hash_password(body.password),
        "created_at": now,
    })

    user_id = str(result.inserted_id)
    token = _create_access_token({"sub": user_id, "email": email})

    debug_info(f"[Auth] Registered: {email}")
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut(id=user_id, name=name, email=email),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login akun",
)
async def login(body: LoginBody) -> TokenResponse:
    """Verifikasi kredensial dan kembalikan JWT token."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    email = body.email.lower().strip()
    user = await db.users.find_one({"email": email})

    # Gunakan pesan generik agar tidak membocorkan info akun mana yang ada
    if not user or not _verify_password(body.password, user["password"]):
        raise HTTPException(status_code=401, detail="Email atau password salah")

    user_id = str(user["_id"])
    token = _create_access_token({"sub": user_id, "email": email})

    debug_info(f"[Auth] Login: {email}")
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut(id=user_id, name=user["name"], email=email),
    )


@router.get(
    "/me",
    response_model=UserOut,
    summary="Info user aktif",
)
async def get_me(token: str = Depends(_oauth2_scheme)) -> UserOut:
    """Kembalikan data user dari JWT token yang aktif."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token tidak valid atau kadaluarsa",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub", "")
        if not user_id:
            raise credentials_error
    except JWTError:
        raise credentials_error

    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise credentials_error

    if not user:
        raise credentials_error

    return UserOut(id=user_id, name=user["name"], email=user["email"])
