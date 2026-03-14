
import io
import logging
import os
import tempfile
import time
from contextlib import suppress
from functools import lru_cache
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from faster_whisper import WhisperModel
from kokoro import KPipeline
from pydantic import BaseModel


def setup_logging() -> logging.Logger:
    level_name = os.getenv("SPEECH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    return logging.getLogger("speech-service")


logger = setup_logging()
app = FastAPI(title="Local Speech Service")

SPEECH_API_KEY = os.getenv("SPEECH_API_KEY", "local-speech-key")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "distil-large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
KOKORO_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")
KOKORO_DEFAULT_VOICE = os.getenv("KOKORO_DEFAULT_VOICE", "af_heart")


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: Optional[float] = 1.0


@lru_cache(maxsize=1)
def get_whisper() -> WhisperModel:
    logger.info(
        "loading_whisper_model model=%s device=%s compute_type=%s",
        WHISPER_MODEL,
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
    )
    return WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


@lru_cache(maxsize=1)
def get_kokoro() -> KPipeline:
    logger.info("loading_kokoro_pipeline lang_code=%s default_voice=%s", KOKORO_LANG_CODE, KOKORO_DEFAULT_VOICE)
    return KPipeline(lang_code=KOKORO_LANG_CODE)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "whisper_model": WHISPER_MODEL,
        "whisper_device": WHISPER_DEVICE,
        "kokoro_voice": KOKORO_DEFAULT_VOICE,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    if authorization and authorization != f"Bearer {SPEECH_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        payload = await file.read()
        tmp.write(payload)
        tmp_path = tmp.name

    whisper = get_whisper()
    started = time.perf_counter()
    try:
        segments, _info = whisper.transcribe(
            tmp_path,
            language=language or None,
            initial_prompt=prompt or None,
            vad_filter=True,
            beam_size=1,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "transcription_completed filename=%s bytes=%s model=%s language=%s chars=%s elapsed_ms=%s",
            file.filename,
            len(payload),
            model,
            language or "auto",
            len(text),
            elapsed_ms,
        )
        return JSONResponse({"text": text, "elapsed_ms": elapsed_ms})
    except Exception:
        logger.exception("transcription_failed filename=%s model=%s", file.filename, model)
        raise
    finally:
        with suppress(Exception):
            os.unlink(tmp_path)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, authorization: Optional[str] = Header(None)):
    if authorization and authorization != f"Bearer {SPEECH_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="input is required")

    voice = req.voice or KOKORO_DEFAULT_VOICE
    pipeline = get_kokoro()
    started = time.perf_counter()

    try:
        audio_chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice, speed=req.speed or 1.0):
            audio_chunks.append(np.asarray(audio, dtype=np.float32))

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="TTS produced no audio")

        waveform = np.concatenate(audio_chunks)
        buffer = io.BytesIO()
        sf.write(buffer, waveform, 24000, format="WAV")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "tts_completed voice=%s chars=%s samples=%s elapsed_ms=%s",
            voice,
            len(text),
            len(waveform),
            elapsed_ms,
        )
        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception:
        logger.exception("tts_failed voice=%s chars=%s", voice, len(text))
        raise
