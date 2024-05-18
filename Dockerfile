# Stage 1
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir -U pip setuptools
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt && find /app/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

# Stage 2
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app /app
ADD data /app/data
COPY gui.py .
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["streamlit", "run", "gui.py", "--server.port=8501", "--server.address=0.0.0.0"]