# Serve image — hybrid twin scoring + RUL (no torch).
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TWIN_MODELS_DIR=/app/service/models

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --create-home app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY service/ service/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[serve]" && \
    rm -rf /root/.cache

USER app

EXPOSE 8000

CMD ["uvicorn", "twin_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
