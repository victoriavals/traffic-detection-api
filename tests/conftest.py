"""
Pytest configuration and shared fixtures for Traffic Counter API tests.

Uses unittest.mock to isolate tests from real MongoDB and GPU resources.
All async tests use pytest-asyncio with asyncio mode = "auto".
"""

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Ensure the api-traffic-counter root is importable ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_db():
    """
    AsyncMock of the Motor database object.

    Provides a mock `video_jobs` collection with common async methods:
    - insert_one, update_one, update_many, find_one, delete_one
    - find (returns an async iterator)

    Usage:
        async def test_something(mock_db):
            with patch("routes.video_jobs.get_db", return_value=mock_db):
                ...
    """
    db = MagicMock()
    collection = MagicMock()

    collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test_id"))
    collection.update_one = AsyncMock(return_value=MagicMock(modified_count=1, upserted_id=None))
    collection.update_many = AsyncMock(return_value=MagicMock(modified_count=0))
    collection.find_one = AsyncMock(return_value=None)
    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

    # find() returns an async generator — mock via __aiter__
    async def _empty_cursor(*args, **kwargs):
        return
        yield  # make it an async generator

    cursor = MagicMock()
    cursor.__aiter__ = lambda self: _empty_cursor()
    collection.find = MagicMock(return_value=cursor)

    db.video_jobs = collection
    return db


@pytest.fixture
def mock_event_loop():
    """
    A real asyncio event loop for testing run_coroutine_threadsafe behaviour.

    Yields the loop and closes it after the test. Tests that call _persist_job
    with this loop will schedule coroutines that can be awaited.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
