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

CMD ["python", "-m", "gradio_app.app"]
