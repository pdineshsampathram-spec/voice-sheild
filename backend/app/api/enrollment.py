"""
Speaker enrollment endpoint.

POST /api/enrollment — Enroll a new speaker voice profile.
"""

import logging
import os
import tempfile
from typing import Optional

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
    speaker_id: Optional[str] = Form(None),
):
    """Enroll a new speaker or update an existing profile with an additional sample.

    The system extracts a normalized speaker embedding and maintains a multi-sample centroid.
    Requires at least 2.0 seconds of audio with usable speech energy.
    """
    if not ecapa_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="ECAPA-TDNN model is not loaded. Speaker enrollment unavailable."
        )

    # Read and save temp file
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="ENROLLMENT_REJECTED: Empty file provided.")

    ext = os.path.splitext(file.filename or "upload.wav")[1].lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load and preprocess (supports WAV, WebM/Opus via PyAV, MP3, OGG, etc.)
        waveform, sr = preprocessor.load_file(tmp_path)
        waveform = preprocessor.preprocess(waveform, sr)

        # Usable speech energy validation
        rms = preprocessor.compute_rms(waveform)
        if rms < 0.002:
            raise HTTPException(
                status_code=400,
                detail="ENROLLMENT_REJECTED: Audio energy too low (silence / ambient noise). Please speak clearly into the microphone."
            )

        # Check minimum duration (~2-3 seconds)
        duration = preprocessor.get_duration(waveform)
        if duration < 2.0:
            raise HTTPException(
                status_code=400,
                detail=f"ENROLLMENT_REJECTED: Audio too short ({duration:.1f}s). Provide at least 3 seconds of speech."
            )

        # Enroll or update running centroid with exact sample reference
        profile = ecapa_service.enroll_speaker(
            speaker_name=speaker_name.strip(),
            waveform=waveform,
            existing_speaker_id=speaker_id.strip() if speaker_id else None,
            sample_bytes=content,
            sample_filename=file.filename or "sample.webm",
            mime_type=file.content_type or "audio/webm",
        )

        return EnrollmentResponse(
            speaker_id=profile.speaker_id,
            speaker_name=profile.speaker_name,
            created_at=profile.created_at,
            sample_count=profile.sample_count,
            quality_status=profile.quality_status,
            message=f"Speaker '{profile.speaker_name}' enrolled successfully (samples: {profile.sample_count}, quality: {profile.quality_status}).",
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


@router.get("/api/enrollment/samples/{sample_id}")
async def get_enrollment_sample(sample_id: str):
    """Serve an exact enrollment voice sample with path traversal protection and HTTP Range support.

    Guarantees:
    - Identifier validated against strict regex
    - File verified to exist within the application's enrollment samples directory
    - Correct audio MIME type returned
    - Returns 404 if missing or unavailable
    """
    import mimetypes
    import re
    from fastapi.responses import FileResponse

    if not sample_id or not re.match(r'^[a-zA-Z0-9_-]+$', sample_id):
        raise HTTPException(status_code=400, detail="Invalid sample identifier.")

    sample_path = ecapa_service.get_sample_path(sample_id)
    if not sample_path or not sample_path.exists():
        raise HTTPException(status_code=404, detail="Enrollment sample unavailable")

    media_type, _ = mimetypes.guess_type(str(sample_path))
    if not media_type:
        ext = sample_path.suffix.lower()
        if ext == ".webm":
            media_type = "audio/webm"
        elif ext == ".wav":
            media_type = "audio/wav"
        elif ext == ".mp3":
            media_type = "audio/mpeg"
        elif ext == ".ogg":
            media_type = "audio/ogg"
        else:
            media_type = "application/octet-stream"

    return FileResponse(
        path=str(sample_path),
        media_type=media_type,
        filename=sample_path.name,
    )


@router.delete("/api/enrollment/samples/{sample_id}")
async def delete_enrollment_sample(sample_id: str):
    """Delete an individual enrollment voice sample artifact and recompute centroid.

    Guarantees:
    - Strict identifier regex validation
    - Filesystem containment checks in ECAPAService
    - Recomputes profile centroid from remaining samples
    - Does not leave a fake VALID profile if all samples are deleted
    - Returns 404 if sample_id is not found
    """
    import re
    if not sample_id or not re.match(r'^[a-zA-Z0-9_-]+$', sample_id):
        raise HTTPException(status_code=400, detail="Invalid sample identifier.")

    updated_profile = ecapa_service.delete_sample(sample_id)
    if not updated_profile:
        raise HTTPException(status_code=404, detail="Enrollment sample not found.")

    return {
        "message": f"Sample '{sample_id}' successfully deleted.",
        "speaker_id": updated_profile.speaker_id,
        "speaker_name": updated_profile.speaker_name,
        "sample_count": updated_profile.sample_count,
        "quality_status": updated_profile.quality_status,
        "profile": updated_profile,
    }


@router.delete("/api/enrollment/{speaker_id}")
async def delete_profile(speaker_id: str):
    """Delete an enrolled speaker profile."""
    if ecapa_service.delete_profile(speaker_id):
        return {"message": f"Profile {speaker_id} deleted."}
    raise HTTPException(status_code=404, detail="Profile not found")

