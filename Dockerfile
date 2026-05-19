FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/cbr \
    CODEBASE_RAG_HOME=/data/codebase-rag \
    OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_NO_CLOUD=1

RUN useradd --create-home --uid 10001 cbr \
    && mkdir -p /app /work /data/codebase-rag \
    && chown -R cbr:cbr /home/cbr /work /data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY codebase_rag ./codebase_rag

RUN pip install --no-cache-dir -e ".[web,serve,tui]"

USER cbr
WORKDIR /work

ENTRYPOINT ["codebase-rag"]
CMD ["doctor", "--root", "/work"]
