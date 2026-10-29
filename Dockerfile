FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md Makefile IMPLEMENTATION_STATUS.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY docs/ ./docs/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps -e .

ENV AI_BACKEND_HOST=0.0.0.0 \
    AI_BACKEND_PORT=8081 \
    AI_DEVICE=auto \
    AI_LOG_LEVEL=INFO \
    RISK_RULES_PATH=config/risk_rules.yaml

EXPOSE 8081

CMD ["python3", "scripts/run_server.py", "--config", "config/default.yaml"]
