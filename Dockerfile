FROM python:3.10-slim

WORKDIR /usr/src/app

RUN apt update
RUN apt install curl -y
RUN apt install iputils-ping -y

COPY requirements.txt ./
RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV LLM_BACKEND="bsc"
ENV PERSISTENT_DATA="/data"
ENV OPENAI_API_ENDPOINT_URLs='["http://localhost:8082/v1,salamandra-7b-instruct,Salamandra (MN5)","http://localhost:8095/v1,salamandra-7b-instruct,Salamandra (AMD)","http://localhost:8086/v1,EuroLLM-9B-Instruct,EuroLLM","http://localhost:8096/v1,Qwen2-Audio-7B-Instruct,Qwen2-Audio"]'

CMD ["bash", "run_script.sh"]
