FROM python:3.10-slim

WORKDIR /usr/src/app

RUN apt update
RUN apt install curl -y
RUN apt install iputils-ping ffmpeg -y

COPY requirements.txt ./
RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV LLM_BACKEND="bsc"
ENV PERSISTENT_DATA="/data"
ENV RETRIEVER_ENDPOINT="http://127.0.0.1:7999"

CMD ["bash", "run_script.sh"]
