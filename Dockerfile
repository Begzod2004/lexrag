FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lexrag ./lexrag
COPY scripts ./scripts
COPY eval ./eval
# data/ (indeks) tom (volume) sifatida ulanadi: -v $(pwd)/data:/app/data
EXPOSE 8000
CMD ["uvicorn", "lexrag.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
