"""
Recorded audio analysis endpoint.

POST /api/analyze/upload — Upload a recording for full analysis.
"""

import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from app.schemas.analysis import AnalysisResult
from app.services.pipeline import analysis_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Supported audio formats
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/api/analyze/upload", response_model=AnalysisResult)
async def analyze_upload(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
):
    """Analyze an uploaded audio recording for voice cloning indicators.

    Supports WAV, MP3, M4A, WebM, OGG, FLAC.
    Optionally provide a speaker_id to verify against an enrolled profile.
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

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = analysis_pipeline.analyze_file(
            file_path=tmp_path,
            speaker_id=speaker_id,
        )
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temp file — don't store raw audio by default
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
