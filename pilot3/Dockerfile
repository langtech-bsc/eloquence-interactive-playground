FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only stack: the LLM runs remotely (OpenAI-compatible endpoint); only LaBSE runs here.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY *.py ./

EXPOSE 8000
