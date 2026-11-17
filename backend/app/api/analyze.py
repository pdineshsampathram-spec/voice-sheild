import hashlib
import json
import logging
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.analysis import AnalysisResult, RecordingReference
from app.services.pipeline import analysis_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Supported audio formats
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def _get_recording_media_type(path: Path) -> str:
    """Resolve correct audio MIME type."""
    media_type, _ = mimetypes.guess_type(str(path))
    if media_type:
        return media_type
    ext = path.suffix.lower()
    mapping = {
        ".webm": "audio/webm",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
    }
    return mapping.get(ext, "application/octet-stream")


def _resolve_recording_file(recording_id: str) -> Optional[Path]:
    """Safely resolve an exact analyzed recording file with strict directory containment."""
    if not recording_id or not re.match(r'^[a-zA-Z0-9_-]+$', recording_id):
        return None

    recordings_dir = settings.recordings_path.resolve()
    for f in recordings_dir.glob(f"{recording_id}.*"):
        if f.is_file() and not f.name.endswith(".json"):
            try:
                resolved = f.resolve()
                if resolved.parent == recordings_dir:
                    return resolved
            except (OSError, ValueError):
                return None
    return None


@router.post("/api/analyze/upload", response_model=AnalysisResult)
async def analyze_upload(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
):
    """Analyze an uploaded audio recording for voice cloning indicators.

    Supports WAV, MP3, M4A, WebM, OGG, FLAC.
    Retains the exact analyzed artifact in recordings_cache for verified playback.
    """
    # Validate file extension
    filename = file.filename or "upload.wav"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {ALLOWED_EXTENSIONS}"
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Cryptographic SHA-256 for exact artifact lineage
    content_sha256 = hashlib.sha256(content).hexdigest()
    recording_id = f"rec_{content_sha256[:16]}"

    # Save to server-managed recordings cache
    recordings_dir = settings.recordings_path
    artifact_path = recordings_dir / f"{recording_id}{ext}"
    meta_path = recordings_dir / f"{recording_id}.json"

    with open(artifact_path, "wb") as f:
        f.write(content)

    media_type = _get_recording_media_type(artifact_path)

    try:
        result = analysis_pipeline.analyze_file(
            file_path=str(artifact_path),
            speaker_id=speaker_id,
        )

        # Attach explicit recording reference to AnalysisResult
        rec_ref = RecordingReference(
            recording_id=recording_id,
            original_filename=filename,
            mime_type=media_type,
            duration_seconds=result.duration_seconds,
            playback_available=True,
            playback_url=f"/api/analyze/recordings/{recording_id}",
            sha256=content_sha256,
        )
        result.recording = rec_ref

        # Store recording metadata
        with open(meta_path, "w") as mf:
            json.dump({
                "recording_id": recording_id,
                "original_filename": filename,
                "mime_type": media_type,
                "duration_seconds": result.duration_seconds,
                "sha256": content_sha256,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, mf)

        return result

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/api/analyze/recordings/{recording_id}")
async def get_analyzed_recording(recording_id: str):
    """Stream the exact analyzed recording artifact with path traversal protection and Range support.

    Guarantees:
    - Recording ID validated against strict alphanumeric regex
    - File verified to reside within server-managed recordings directory
    - Prevents directory traversal attacks
    - Returns 404 if file was deleted or no longer exists
    - Never guesses or silently substitutes another recording
    """
    if not recording_id or not re.match(r'^[a-zA-Z0-9_-]+$', recording_id):
        raise HTTPException(status_code=400, detail="Invalid recording identifier.")

    resolved_path = _resolve_recording_file(recording_id)
    if not resolved_path or not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Recording no longer available")

    # Read original filename if available from metadata
    recordings_dir = settings.recordings_path
    meta_path = recordings_dir / f"{recording_id}.json"
    download_filename = resolved_path.name
    if meta_path.exists():
        try:
            with open(meta_path) as mf:
                meta = json.load(mf)
                download_filename = meta.get("original_filename", resolved_path.name)
        except Exception:
            pass

    media_type = _get_recording_media_type(resolved_path)

    return FileResponse(
        path=str(resolved_path),
        media_type=media_type,
        filename=download_filename,
    )


@router.delete("/api/analyze/recordings/{recording_id}")
async def delete_analyzed_recording(recording_id: str):
    """Delete an analyzed recording artifact (for testing unavailable state and cleanup)."""
    if not recording_id or not re.match(r'^[a-zA-Z0-9_-]+$', recording_id):
        raise HTTPException(status_code=400, detail="Invalid recording identifier.")

    resolved_path = _resolve_recording_file(recording_id)
    meta_path = settings.recordings_path / f"{recording_id}.json"

    deleted = False
    if resolved_path and resolved_path.exists():
        resolved_path.unlink()
        deleted = True
    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    if deleted:
        return {"message": f"Recording {recording_id} removed."}
    raise HTTPException(status_code=404, detail="Recording not found")
