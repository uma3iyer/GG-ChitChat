FROM python:3.11-slim

# Route all model / Hugging Face caches to a writable dir (Spaces runs non-root; /tmp is writable)
ENV HF_HOME=/tmp/hf \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    HF_HUB_CACHE=/tmp/hf \
    PYTHONUNBUFFERED=1

# Non-root user (Hugging Face Spaces requirement; UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app
WORKDIR /home/user/app

# Install dependencies first (better layer caching)
COPY --chown=user:user requirements-api.txt .
RUN pip install --no-cache-dir --user -r requirements-api.txt

# App code + prebuilt style artifacts
COPY --chown=user:user api ./api
COPY --chown=user:user src ./src
COPY --chown=user:user data/processed ./data/processed

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]