FROM python:3.10-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV LLM_BACKEND="bsc"
ENV OPENAI_API_ENDPOINT_URL="http://172.17.0.1:8081/v1"

CMD ["python", "-m", "gradio_app.app"]
