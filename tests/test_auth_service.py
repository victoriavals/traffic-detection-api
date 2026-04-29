"""Tests for password hashing helpers."""

from services.auth_service import hash_password, verify_password


def test_hash_password_verifies_ascii_password() -> None:
    hashed = hash_password("CorrectHorse123")

    assert verify_password("CorrectHorse123", hashed)
    assert not verify_password("WrongHorse123", hashed)


def test_hash_password_handles_passwords_over_72_bytes() -> None:
    # 30 emojis are 120 UTF-8 bytes. bcrypt only accepts 72 bytes, so the
    # helper must truncate before calling bcrypt instead of letting bcrypt raise.
    password = "\U0001f512" * 30
    hashed = hash_password(password)

    assert verify_password(password, hashed)


def test_verify_password_returns_false_for_invalid_hash() -> None:
    assert not verify_password("CorrectHorse123", "not-a-bcrypt-hash")
