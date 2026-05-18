"""
Authentication routes — register, login, refresh, me.

All endpoints under ``/auth``. JWT-based with separate access (15 min) and
refresh (7 day) tokens. First registered user becomes admin; rest default to
operator.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError
from pymongo.errors import DuplicateKeyError

from constant_var import debug_info, debug_warning
from dependencies.auth import get_current_user
from models.schemas import (
    AccessTokenOnly,
    RefreshRequest,
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
)
from services.auth_service import (
    InviteCodeNotFound,
    create_access_token,
    create_refresh_token,
    create_user,
    decode_token,
    get_user_by_email,
    user_to_response,
    verify_password,
)
from services.database import get_db

router = APIRouter()


@router.post(
    "/auth/register",
    response_model=Token,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account. First user becomes admin; subsequent users default to operator. Returns access + refresh tokens for auto-login.",
    tags=["🔐 Auth"],
)
async def register(payload: UserCreate) -> Token:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Nama tidak boleh kosong atau hanya spasi")

    invite_code: str | None = None
    if payload.invite_code:
        invite_code = payload.invite_code.strip().upper()
        if not invite_code:
            invite_code = None

    try:
        user = await create_user(payload.email, name, payload.password, invite_code=invite_code)
    except InviteCodeNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Email sudah terdaftar")

    join_log = f" (joined org via invite_code)" if invite_code else " (new org)"
    debug_info(f"[Auth] New user registered: {user['email']} (role={user['role']}){join_log}")

    return Token(
        access_token=create_access_token(user["email"], user["role"]),
        refresh_token=create_refresh_token(user["email"]),
        user=UserResponse(**user_to_response(user)),
    )


@router.post(
    "/auth/login",
    response_model=Token,
    summary="Login with email & password",
    description="Authenticate and return access + refresh tokens.",
    tags=["🔐 Auth"],
)
async def login(payload: UserLogin) -> Token:
    user = await get_user_by_email(payload.email)
    # Generic 401 for invalid creds, missing user, OR disabled — no enumeration
    if (
        not user
        or user.get("status", "active") == "disabled"
        or not verify_password(payload.password, user["password_hash"])
    ):
        debug_warning(f"[Auth] Failed login attempt: {payload.email}")
        raise HTTPException(status_code=401, detail="Email atau password salah")

    debug_info(f"[Auth] Login: {user['email']}")

    return Token(
        access_token=create_access_token(user["email"], user["role"]),
        refresh_token=create_refresh_token(user["email"]),
        user=UserResponse(**user_to_response(user)),
    )


@router.post(
    "/auth/refresh",
    response_model=AccessTokenOnly,
    summary="Refresh access token",
    description="Exchange a valid refresh token for a new access token.",
    tags=["🔐 Auth"],
)
async def refresh(payload: RefreshRequest) -> AccessTokenOnly:
    try:
        decoded = decode_token(payload.refresh_token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    if decoded.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Token bukan refresh token")

    email = decoded.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Refresh token payload tidak valid")

    user = await get_user_by_email(email)
    if not user or user.get("status", "active") == "disabled":
        raise HTTPException(status_code=401, detail="User tidak ditemukan")

    return AccessTokenOnly(access_token=create_access_token(user["email"], user["role"]))


@router.get(
    "/auth/me",
    response_model=UserResponse,
    summary="Get current authenticated user",
    description="Return profile of the user owning the supplied access token.",
    tags=["🔐 Auth"],
)
async def me(user: dict = Depends(get_current_user)) -> UserResponse:
    return UserResponse(**user_to_response(user))
