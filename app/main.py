"""
Whisper Transcription API - Dialogy
====================================
API de transcrição de áudio usando faster-whisper (large-v3-turbo)
Otimizada para CPU com compute_type int8.
"""

import os
import time
import uuid
import logging
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("whisper-api")

# ---------------------------------------------------------------------------
# Configuração via variáveis de ambiente
# ---------------------------------------------------------------------------
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
API_KEY = os.getenv("WHISPER_API_KEY", "")  # vazio = sem autenticação

# ---------------------------------------------------------------------------
# Inicialização do modelo (singleton – carrega uma vez na startup)
# ---------------------------------------------------------------------------
model: Optional[WhisperModel] = None


def load_model() -> WhisperModel:
    """Carrega o modelo Whisper na memória."""
    logger.info(
        "Carregando modelo '%s' | device=%s | compute_type=%s | threads=%d",
        MODEL_SIZE, DEVICE, COMPUTE_TYPE, CPU_THREADS,
    )
    start = time.time()
    m = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        num_workers=NUM_WORKERS,
    )
    elapsed = time.time() - start
    logger.info("Modelo carregado em %.2fs", elapsed)
    return m


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Whisper Transcription API",
    description="API de transcrição de áudio via faster-whisper (large-v3-turbo). Deploy Dialogy.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware de autenticação (opcional)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if API_KEY and request.url.path not in ("/", "/health", "/docs", "/redoc", "/openapi.json"):
        token = request.headers.get("Authorization", "")
        if token != f"Bearer {API_KEY}":
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


# ---------------------------------------------------------------------------
# Eventos de lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    compute_type: str


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    request_id: str
    text: str
    language: str
    language_probability: float
    duration_seconds: float
    processing_time_seconds: float
    segments: list[TranscriptionSegment]


class DetectLanguageResponse(BaseModel):
    language: str
    language_probability: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
async def root():
    """Informações básicas da API."""
    return {
        "service": "Whisper Transcription API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    """Health check – confirma que o modelo está carregado."""
    return HealthResponse(
        status="healthy" if model else "loading",
        model=MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )


@app.post("/v1/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe(
    file: UploadFile = File(..., description="Arquivo de áudio (mp3, wav, m4a, ogg, flac, webm …)"),
    language: Optional[str] = Form(None, description="Código ISO do idioma (ex: pt, en). Auto-detect se vazio."),
    task: str = Form("transcribe", description="'transcribe' ou 'translate' (traduz para inglês)"),
    initial_prompt: Optional[str] = Form(None, description="Prompt inicial para guiar a transcrição"),
    word_timestamps: bool = Form(False, description="Incluir timestamps por palavra"),
    vad_filter: bool = Form(True, description="Filtrar silêncios via VAD (Voice Activity Detection)"),
    beam_size: int = Form(5, description="Beam size para decodificação"),
):
    """
    Transcreve um arquivo de áudio.

    Aceita os formatos: mp3, wav, m4a, ogg, flac, webm, mp4, mpeg, mpga, oga, opus.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda carregando. Tente novamente em instantes.")

    # Validar tamanho
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande ({size_mb:.1f}MB). Limite: {MAX_FILE_SIZE_MB}MB.",
        )

    request_id = str(uuid.uuid4())
    logger.info("[%s] Recebido: %s (%.2fMB) | lang=%s task=%s", request_id, file.filename, size_mb, language, task)

    # Salvar temporariamente
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        start_time = time.time()

        segments_raw, info = model.transcribe(
            tmp_path,
            language=language,
            task=task,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )

        # Materializar segmentos
        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []
        for i, seg in enumerate(segments_raw):
            segments.append(TranscriptionSegment(
                id=i,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
            ))
            full_text_parts.append(seg.text.strip())

        processing_time = time.time() - start_time
        full_text = " ".join(full_text_parts)

        logger.info(
            "[%s] Concluído em %.2fs | lang=%s (%.0f%%) | %d segmentos",
            request_id, processing_time, info.language, info.language_probability * 100, len(segments),
        )

        return TranscriptionResponse(
            request_id=request_id,
            text=full_text,
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration_seconds=round(info.duration, 3),
            processing_time_seconds=round(processing_time, 3),
            segments=segments,
        )

    except Exception as e:
        logger.error("[%s] Erro na transcrição: %s", request_id, str(e))
        raise HTTPException(status_code=500, detail=f"Erro na transcrição: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/v1/detect-language", response_model=DetectLanguageResponse, tags=["Transcription"])
async def detect_language(
    file: UploadFile = File(..., description="Arquivo de áudio para detecção de idioma"),
):
    """Detecta o idioma do áudio enviado."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda carregando.")

    contents = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=1)
        # Precisamos consumir pelo menos um segmento para o info ser populado
        for _ in segments:
            break
        return DetectLanguageResponse(
            language=info.language,
            language_probability=round(info.language_probability, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na detecção: {str(e)}")
    finally:
        os.unlink(tmp_path)
