from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import base64
from pydantic import BaseModel

from gradio_app.backend.query_llm import LLMHandler

llm_handler = LLMHandler()
app = FastAPI()

buffer = []


class Request(BaseModel):
    history: list[list[str]]
    llm_name: str
    query: str


class Response(BaseModel):
    text: str
    documents: list[str]


def encode_audio_stream(audio):
    try:
        audio = bytes(audio)
        encoded = base64.b64encode(audio).decode("utf-8")
        return encoded
    except:
        return ""


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    global buffer
    while True:
        try:
            data = await websocket.receive_bytes()
            buffer.extend(data)
            # Add real-time processing here
        except WebSocketDisconnect:
            print("Disconnected")
            print("Buffer size", len(buffer))
        except Exception as e:
            print("Connection closed:", e)
            break

@app.post("/respond", response_model=Response)
async def respond(request: Request):
# def respond_with_text(llm_name, system_prompt, history, documents):
    global buffer
    response = []
    for part in llm_handler(request.llm_name, "", request.history, [], audio=encode_audio_stream(buffer)):
        response.extend(part)
    buffer = []
    return Response(
        text="".join(response),
        documents=[]
    )

if __name__ == "__main__":
    uvicorn.run("gradio_app.backend.ws_audio_server:app", host="0.0.0.0", port=8999, reload=True)
