FROM python:3.10-slim

WORKDIR /usr/src/app

RUN apt update
RUN apt install curl -y
RUN apt install iputils-ping ffmpeg -y

COPY requirements.txt ./
RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install -c requirements.txt torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --retries 10 --timeout 120 -c requirements.txt FlagEmbedding==1.3.5

RUN pip install -c requirements.txt openai==1.102.0

COPY vendor/eloquence_orchestrator-0.1.0-py3-none-any.whl /opt/wheels/
RUN pip install --no-cache-dir -c requirements.txt /opt/wheels/eloquence_orchestrator-0.1.0-py3-none-any.whl
RUN python -c "from eloquence_orchestrator.orchestrator import build_graph_ip; from eloquence_orchestrator.mcp_tools import call_mcp_tool"
RUN python -c "import gradio; from huggingface_hub import HfFolder; from FlagEmbedding import FlagReranker"

COPY . .

EXPOSE 8080          
EXPOSE 7860
EXPOSE 7997
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV LLM_BACKEND="bsc"
ENV PERSISTENT_DATA="/data"
ENV RETRIEVER_ENDPOINT="http://0.0.0.0:7997"

CMD ["bash", "run_script.sh"]
