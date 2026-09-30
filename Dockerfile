# ---- Tầng 1: build React ----
FROM node:22-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- Tầng 2: API Python + mô hình ----
FROM python:3.11-slim
# Container của HF Spaces chạy bằng UID 1000, nên tạo user trước mọi COPY để dùng --chown.
# (chown -R sau COPY sẽ nhân bản layer: image phình thêm ~100 MB vì artifacts + gallery.)
RUN useradd -m -u 1000 app
WORKDIR /app
RUN chown app:app /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app config.py ./
COPY --chown=app:app core/ core/
COPY --chown=app:app api/ api/
COPY --chown=app:app data/kb data/kb
COPY --chown=app:app data/gallery data/gallery
COPY --chown=app:app artifacts/ artifacts/
COPY --from=web --chown=app:app /web/dist web/dist
ENV APP_ROOT=/app HF_HOME=/app/.cache LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
USER app
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
