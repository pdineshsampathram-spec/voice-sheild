"""
Speaker enrollment endpoint.

POST /api/enrollment — Enroll a new speaker voice profile.
"""

import logging
import os
import tempfile

from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from app.schemas.analysis import EnrollmentResponse, SpeakerProfile
from app.services.speaker.ecapa_service import ecapa_service
from app.services.audio.preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)
router = APIRouter()

preprocessor = AudioPreprocessor()


@router.post("/api/enrollment", response_model=EnrollmentResponse)
async def enroll_speaker(
    file: UploadFile = File(...),
    speaker_name: str = Form(...),
):
    """Enroll a new speaker by uploading a voice sample.

    The system extracts a speaker embedding and stores it as a profile.
    Provide a clear recording of at least 5 seconds of speech.
    """
    if not ecapa_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="ECAPA-TDNN model is not loaded. Speaker enrollment unavailable."
        )

    # Read and save temp file
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    ext = os.path.splitext(file.filename or "upload.wav")[1].lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load and preprocess
        waveform, sr = preprocessor.load_file(tmp_path)
        waveform = preprocessor.preprocess(waveform, sr)

        # Check minimum duration (~3 seconds)
        duration = preprocessor.get_duration(waveform)
        if duration < 2.0:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short ({duration:.1f}s). Provide at least 3 seconds of speech."
            )

        # Enroll
        profile = ecapa_service.enroll_speaker(speaker_name, waveform)

        return EnrollmentResponse(
            speaker_id=profile.speaker_id,
            speaker_name=profile.speaker_name,
            created_at=profile.created_at,
            message=f"Speaker '{speaker_name}' enrolled successfully.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/api/enrollment/profiles", response_model=list[SpeakerProfile])
async def list_profiles():
    """List all enrolled speaker profiles."""
    return ecapa_service.list_profiles()


@router.delete("/api/enrollment/{speaker_id}")
async def delete_profile(speaker_id: str):
    """Delete an enrolled speaker profile."""
    if ecapa_service.delete_profile(speaker_id):
        return {"message": f"Profile {speaker_id} deleted."}
    raise HTTPException(status_code=404, detail="Profile not found")
