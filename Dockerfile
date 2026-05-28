# ===========================================================================
# Whisper Transcription API – Dialogy
# ===========================================================================
# Imagem Docker otimizada para CPU com faster-whisper (large-v3-turbo)
# ===========================================================================

FROM python:3.11-slim AS base

# Evitar prompts interativos e definir locale
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dependências de sistema para áudio (ffmpeg é obrigatório para whisper)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
WORKDIR /app

# Instalar dependências Python primeiro (cache de layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY app/ ./app/

# ---------------------------------------------------------------------------
# Pré-download do modelo (para não baixar na primeira request)
# Isso aumenta o tamanho da imagem (~3GB), mas garante cold-start rápido.
# Comente as linhas abaixo se preferir download on-demand.
# ---------------------------------------------------------------------------
RUN python -c "\
from faster_whisper import WhisperModel; \
print('Baixando modelo large-v3-turbo...'); \
WhisperModel('large-v3-turbo', device='cpu', compute_type='int8'); \
print('Modelo baixado com sucesso!')"

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 8123

# Health check nativo do Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8123/health || exit 1

# Rodar com uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8123", "--workers", "1"]
