# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    TRANSCRIPTION_BACKEND=local_edge \
    KB_WHISPER_SIZE=large

WORKDIR /app

# Install the application and the FastAPI/local-model runtime. Dependency wheels
# are removed after installation to keep the resulting image smaller.
COPY pyproject.toml setup.py _editable_build_backend.py README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir '.[edge]'

COPY data ./data
COPY outputs ./outputs

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/edge_cases \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/health', timeout=3)"

CMD ["sh", "-c", "ambulance-case serve-edge --host 0.0.0.0 --port ${PORT:-8080} --transcription-backend ${TRANSCRIPTION_BACKEND:-local_edge} --kb-whisper-size ${KB_WHISPER_SIZE:-large}"]
