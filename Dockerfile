FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.state/hf
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
RUN useradd -m rag && mkdir -p /app/.state /app/data && chown -R rag /app
USER rag
EXPOSE 8000 8501
# default = API; the compose file overrides the command for the UI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
