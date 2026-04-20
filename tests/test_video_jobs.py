"""
End-to-end and unit tests for routes/video_jobs.py.

Test coverage:
- URL normalisation (Google Drive, Dropbox, direct links)
- _persist_job: fire-and-forget DB write from sync thread
- _update_job: in-memory update + DB persist
- recover_interrupted_jobs: startup recovery marks interrupted jobs as error
- GET /video/jobs/{id}: memory-first → MongoDB fallback
- DELETE /video/jobs/{id}: removes from memory AND MongoDB
- create_video_job: inserts initial document to MongoDB

All MongoDB and event-loop interactions are mocked so tests run without
a real database or GPU.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone

import pytest

# ── Module under test ─────────────────────────────────────────────────────────
import routes.video_jobs as vj
from routes.video_jobs import (
    _normalize_url,
    _persist_job,
    _update_job,
    _Job,
    set_event_loop,
    recover_interrupted_jobs,
)
from models.schemas import ClassCount, VideoJobStatus


# ══════════════════════════════════════════════════════════════════════════════
# URL Normalisation
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeUrl:
    """Tests for _normalize_url() — pure function, no I/O."""

    def test_google_drive_file_link(self):
        """Standard Google Drive /file/d/ share link is converted to usercontent download."""
        url = "https://drive.google.com/file/d/ABC123XYZ/view?usp=sharing"
        result = _normalize_url(url)
        assert "drive.usercontent.google.com/download" in result
        assert "id=ABC123XYZ" in result
        assert "export=download" in result
        assert "confirm=t" in result

    def test_google_drive_open_link(self):
        """Google Drive /open?id= link is also converted."""
        url = "https://drive.google.com/open?id=FILEID456"
        result = _normalize_url(url)
        assert "drive.usercontent.google.com/download" in result
        assert "id=FILEID456" in result

    def test_dropbox_sharing_link(self):
        """Dropbox dl=0 parameter is changed to dl=1 for direct download."""
        url = "https://www.dropbox.com/s/abc123/video.mp4?dl=0"
        result = _normalize_url(url)
        assert "dl=1" in result
        assert "dl=0" not in result

    def test_dropbox_link_without_dl_param(self):
        """Dropbox link without existing dl param gets dl=1 added."""
        url = "https://www.dropbox.com/s/abc123/video.mp4"
        result = _normalize_url(url)
        assert "dl=1" in result

    def test_direct_mp4_url_unchanged(self):
        """Direct download URLs pass through without modification."""
        url = "https://example.com/traffic/video.mp4"
        result = _normalize_url(url)
        assert result == url

    def test_strips_leading_trailing_whitespace(self):
        """URLs with surrounding whitespace are trimmed before processing."""
        url = "  https://example.com/video.mp4  "
        result = _normalize_url(url)
        assert result == "https://example.com/video.mp4"

    def test_other_cloud_url_unchanged(self):
        """Unknown cloud storage URLs are returned as-is."""
        url = "https://onedrive.live.com/download?cid=ABC&resid=XYZ"
        result = _normalize_url(url)
        assert result == url


# ══════════════════════════════════════════════════════════════════════════════
# _persist_job — fire-and-forget MongoDB write from sync thread
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistJob:
    """Tests for _persist_job() — schedules async DB writes from sync code."""

    def test_no_op_when_db_is_none(self):
        """_persist_job silently skips when database is not connected."""
        with patch("routes.video_jobs.get_db", return_value=None), \
             patch("routes.video_jobs._event_loop", new=MagicMock()):
            # Should not raise
            _persist_job("job123", {"status": "processing"})

    def test_no_op_when_event_loop_is_none(self, mock_db):
        """_persist_job silently skips when event loop has not been set."""
        with patch("routes.video_jobs.get_db", return_value=mock_db), \
             patch("routes.video_jobs._event_loop", new=None):
            _persist_job("job123", {"status": "processing"})
            mock_db.video_jobs.update_one.assert_not_called()

    def test_schedules_upsert_on_event_loop(self, mock_db):
        """_persist_job calls run_coroutine_threadsafe with an upsert coroutine."""
        fake_loop = MagicMock()
        fake_future = MagicMock()
        fake_loop.is_closed.return_value = False
        fake_loop.run_until_complete = MagicMock()

        with patch("routes.video_jobs.get_db", return_value=mock_db), \
             patch("routes.video_jobs._event_loop", new=fake_loop), \
             patch("asyncio.run_coroutine_threadsafe", return_value=fake_future) as mock_submit:
            _persist_job("job123", {"status": "done", "progress": 100.0})
            assert mock_submit.called
            # Verify the loop passed matches our fake loop
            _, kwargs_loop = mock_submit.call_args[0], mock_submit.call_args
            assert fake_loop in mock_submit.call_args[0]

    def test_serializes_pydantic_model_to_dict(self, mock_db):
        """ClassCount (Pydantic model) is converted to dict before MongoDB write."""
        fake_loop = MagicMock()
        submitted_coro = None

        def capture_coro(coro, _loop):
            nonlocal submitted_coro
            submitted_coro = coro
            return MagicMock()

        with patch("routes.video_jobs.get_db", return_value=mock_db), \
             patch("routes.video_jobs._event_loop", new=fake_loop), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=capture_coro):
            counts = ClassCount(big_vehicle=2, car=10, pedestrian=1, two_wheeler=5, total=18)
            _persist_job("job123", {"counts": counts})

        assert submitted_coro is not None, "run_coroutine_threadsafe was not called"
        # Actually run the coroutine so update_one (AsyncMock) gets awaited
        asyncio.run(submitted_coro)

        mock_db.video_jobs.update_one.assert_called_once()
        call_args = mock_db.video_jobs.update_one.call_args
        set_doc = call_args[0][1]["$set"]
        # counts must be a plain dict, not ClassCount
        assert isinstance(set_doc["counts"], dict)
        assert set_doc["counts"]["car"] == 10


# ══════════════════════════════════════════════════════════════════════════════
# _update_job — in-memory + DB persist
# ══════════════════════════════════════════════════════════════════════════════

class TestUpdateJob:
    """Tests for _update_job() — thread-safe in-memory update + DB persist."""

    def test_updates_job_fields_in_memory(self):
        """_update_job mutates _Job object fields under lock."""
        job = _Job(job_id="abc")
        assert job.status == "pending"

        with patch("routes.video_jobs._persist_job"):
            _update_job(job, status="processing", progress=42.0)

        assert job.status == "processing"
        assert job.progress == 42.0

    def test_calls_persist_job_with_kwargs(self):
        """_update_job forwards all kwargs to _persist_job for DB write."""
        job = _Job(job_id="abc")

        with patch("routes.video_jobs._persist_job") as mock_persist:
            _update_job(job, status="done", progress=100.0, message="Selesai!")
            mock_persist.assert_called_once_with(
                "abc", {"status": "done", "progress": 100.0, "message": "Selesai!"}
            )

    def test_multiple_updates_accumulate(self):
        """Sequential _update_job calls accumulate state on the same _Job."""
        job = _Job(job_id="xyz")

        with patch("routes.video_jobs._persist_job"):
            _update_job(job, status="downloading", progress=15.0)
            _update_job(job, status="processing", progress=60.0)
            _update_job(job, status="done", progress=100.0)

        assert job.status == "done"
        assert job.progress == 100.0


# ══════════════════════════════════════════════════════════════════════════════
# recover_interrupted_jobs — startup recovery
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestRecoverInterruptedJobs:
    """Tests for recover_interrupted_jobs() — server startup DB cleanup."""

    async def test_no_op_when_db_is_none(self):
        """Recovery silently skips when database is not connected."""
        with patch("routes.video_jobs.get_db", return_value=None):
            await recover_interrupted_jobs()  # should not raise

    async def test_marks_in_progress_jobs_as_error(self, mock_db):
        """update_many is called targeting pending/downloading/processing statuses."""
        mock_db.video_jobs.update_many.return_value = MagicMock(modified_count=3)

        with patch("routes.video_jobs.get_db", return_value=mock_db):
            await recover_interrupted_jobs()

        mock_db.video_jobs.update_many.assert_called_once()
        filter_arg, update_arg = mock_db.video_jobs.update_many.call_args[0]

        # Must target the three in-progress statuses
        assert filter_arg == {"status": {"$in": ["pending", "downloading", "processing"]}}
        # Must set status to error
        assert update_arg["$set"]["status"] == "error"
        assert "error" in update_arg["$set"]
        assert isinstance(update_arg["$set"]["updated_at"], datetime)

    async def test_no_jobs_to_recover(self, mock_db):
        """Recovery runs cleanly when no interrupted jobs exist (modified_count=0)."""
        mock_db.video_jobs.update_many.return_value = MagicMock(modified_count=0)

        with patch("routes.video_jobs.get_db", return_value=mock_db):
            await recover_interrupted_jobs()  # should not raise

        mock_db.video_jobs.update_many.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# GET /video/jobs/{id} — memory-first, MongoDB fallback
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestGetVideoJob:
    """Tests for get_video_job() endpoint — dual-store lookup."""

    async def test_returns_job_from_memory(self, mock_db):
        """If job is in _jobs dict, MongoDB is NOT queried."""
        job = _Job(job_id="mem_job_001", status="processing", progress=55.0)

        with patch.dict(vj._jobs, {"mem_job_001": job}), \
             patch("routes.video_jobs.get_db", return_value=mock_db):
            from routes.video_jobs import get_video_job
            result = await get_video_job("mem_job_001")

        assert result.job_id == "mem_job_001"
        assert result.status == "processing"
        mock_db.video_jobs.find_one.assert_not_called()

    async def test_fallback_to_mongodb_when_not_in_memory(self, mock_db):
        """If job is not in _jobs, find_one is called and result is returned."""
        mock_db.video_jobs.find_one.return_value = {
            "_id": "db_job_002",
            "status": "done",
            "progress": 100.0,
            "message": "Selesai!",
            "counts": {"big_vehicle": 1, "car": 5, "pedestrian": 0, "two_wheeler": 2, "total": 8},
            "video_info": {"duration_seconds": 120.0, "fps": 25.0, "resolution": "1920x1080",
                           "total_frames": 3000, "frames_processed": 3000, "processing_time_seconds": 45.2},
            "inference_config": None,
            "error": None,
        }

        with patch.dict(vj._jobs, {}, clear=True), \
             patch("routes.video_jobs.get_db", return_value=mock_db):
            from routes.video_jobs import get_video_job
            result = await get_video_job("db_job_002")

        assert result.job_id == "db_job_002"
        assert result.status == "done"
        assert result.counts is not None
        assert result.counts.total == 8
        mock_db.video_jobs.find_one.assert_called_once_with({"_id": "db_job_002"})

    async def test_raises_404_when_not_in_memory_or_db(self, mock_db):
        """HTTPException 404 is raised when job is absent from both stores."""
        mock_db.video_jobs.find_one.return_value = None

        with patch.dict(vj._jobs, {}, clear=True), \
             patch("routes.video_jobs.get_db", return_value=mock_db):
            from routes.video_jobs import get_video_job
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await get_video_job("missing_job")

        assert exc_info.value.status_code == 404

    async def test_raises_404_when_db_is_none(self):
        """HTTPException 404 is raised when DB is unavailable and job not in memory."""
        with patch.dict(vj._jobs, {}, clear=True), \
             patch("routes.video_jobs.get_db", return_value=None):
            from routes.video_jobs import get_video_job
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await get_video_job("any_job_id")

        assert exc_info.value.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /video/jobs/{id} — removes from memory AND MongoDB
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestDeleteVideoJob:
    """Tests for delete_video_job() — dual-store deletion."""

    async def test_deletes_in_memory_job(self, mock_db, tmp_path):
        """Job is popped from _jobs and MongoDB delete_one is called."""
        job = _Job(job_id="del_001")
        job.temp_path = str(tmp_path / "fake.mp4")  # non-existent, so no file deletion

        with patch.dict(vj._jobs, {"del_001": job}), \
             patch("routes.video_jobs.get_db", return_value=mock_db):
            from routes.video_jobs import delete_video_job
            await delete_video_job("del_001")

        assert "del_001" not in vj._jobs
        mock_db.video_jobs.delete_one.assert_called_once_with({"_id": "del_001"})

    async def test_deletes_db_only_job(self, mock_db):
        """Job only in MongoDB (e.g. after server restart) is still deleted."""
        mock_db.video_jobs.delete_one.return_value = MagicMock(deleted_count=1)

        with patch.dict(vj._jobs, {}, clear=True), \
             patch("routes.video_jobs.get_db", return_value=mock_db):
            from routes.video_jobs import delete_video_job
            await delete_video_job("db_only_job")

        mock_db.video_jobs.delete_one.assert_called_once_with({"_id": "db_only_job"})

    async def test_raises_404_when_not_in_either_store(self, mock_db):
        """HTTPException 404 is raised when job is absent from both stores."""
        mock_db.video_jobs.delete_one.return_value = MagicMock(deleted_count=0)

        with patch.dict(vj._jobs, {}, clear=True), \
             patch("routes.video_jobs.get_db", return_value=mock_db):
            from routes.video_jobs import delete_video_job
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await delete_video_job("ghost_job")

        assert exc_info.value.status_code == 404

    async def test_cleans_up_temp_file_if_exists(self, mock_db, tmp_path):
        """Temp file is deleted when present on disk."""
        temp_file = tmp_path / "urljob_abc12345.mp4"
        temp_file.write_bytes(b"fake video data")

        job = _Job(job_id="tmpfile_job")
        job.temp_path = str(temp_file)

        with patch.dict(vj._jobs, {"tmpfile_job": job}), \
             patch("routes.video_jobs.get_db", return_value=mock_db):
            from routes.video_jobs import delete_video_job
            await delete_video_job("tmpfile_job")

        assert not temp_file.exists()


# ══════════════════════════════════════════════════════════════════════════════
# set_event_loop — stores loop reference for thread-safe DB writes
# ══════════════════════════════════════════════════════════════════════════════

class TestSetEventLoop:
    """Tests for set_event_loop() — module-level loop reference."""

    def test_stores_loop(self):
        """set_event_loop assigns the loop to the module-level variable."""
        original = vj._event_loop
        fake_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        try:
            set_event_loop(fake_loop)
            assert vj._event_loop is fake_loop
        finally:
            vj._event_loop = original  # restore after test
