"""
Auth dependencies for FastAPI routes.

Provides:
* ``get_current_user`` — extract & validate Bearer token, return user dict.
* ``require_admin`` — same as above, but rejects non-admin users with 403.

These dependencies are not yet attached to any existing endpoints (image,
video, stats, etc.) — that gating happens in a follow-up phase to avoid
breaking external consumers. They are ready to be added via
``Depends(get_current_user)`` to protect future endpoints.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from services.auth_service import decode_token, get_user_by_email

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict:
    """Validate access token and return the user document."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = await get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Same as ``get_current_user`` but only allows admin role."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
