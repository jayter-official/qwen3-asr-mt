# Multi-tenant streaming ASR server for Qwen3-ASR.
#
# We build on top of the upstream qwen3-asr image (which already ships vLLM +
# qwen-asr + CUDA runtime) and layer our FastAPI server on top. This keeps the
# image small and avoids re-compiling vLLM.
ARG BASE_IMAGE=qwenllm/qwen3-asr:latest
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="qasr-mt"
LABEL org.opencontainers.image.description="Multi-tenant streaming ASR server for Qwen3-ASR"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN pip install --no-cache-dir --break-system-packages \
      fastapi \
      'uvicorn[standard]'

WORKDIR /app
COPY qasr_server /app/qasr_server

ENV QASR_MODEL="Qwen/Qwen3-ASR-0.6B" \
    QASR_HOST="0.0.0.0" \
    QASR_PORT="8000" \
    QASR_CHUNK_SIZE_SEC="1.0" \
    QASR_UNFIXED_CHUNK_NUM="4" \
    QASR_UNFIXED_TOKEN_NUM="5" \
    QASR_MAX_NEW_TOKENS="32" \
    QASR_GPU_MEM_UTIL="0.35" \
    QASR_MAX_MODEL_LEN="8192" \
    QASR_SESSION_TTL_SEC="600"

EXPOSE 8000

# HF cache should be mounted for faster cold starts; model downloads on first run otherwise.
VOLUME ["/root/.cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python3 -c "import urllib.request,sys,json; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3); d=json.load(r); sys.exit(0 if d.get('ready') else 1)" || exit 1

CMD ["bash", "/app/qasr_server/launcher.sh"]
