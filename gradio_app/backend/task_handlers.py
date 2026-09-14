import re

import requests
from gradio_app.helpers import check_llm_interface

def get_task_handler(config, llm, retriver):
    if config["service"].startswith("pilot3"):
        return Pilot3TaskHandler(task_config=config)
    if "local" in config["service"]:
        return LocalTaskHandler(llm_handler=llm, retriever=retriver, task_config=config)
    if "remote" in config["service"]:
        return RemoteTaskHandler(task_config=config)
    

class LocalTaskHandler:
    def __init__(self, llm_handler, retriever, task_config):
        self.llm_handler = llm_handler
        self.retriever = retriever
        self.task_config = task_config
        
    def __call__(self, llm_name, system_prompt, history, query, docs_k, index_name, **params):
        if not check_llm_interface(llm_name, self.task_config["interface"], available_llms=self.llm_handler.available_llms):
            raise ValueError(f"LLM {llm_name} does not support the required interface {self.task_config['interface']}.")
        if self.task_config.get("interface") == "audio":
            audio_mode = self.task_config.get("audio_mode")
            if audio_mode:
                entry = self.llm_handler.available_llms.get(llm_name, {})
                interactor = str(entry.get("interactor", "")).strip().lower()

                is_whisper = interactor == "whisper"
                is_whisperx = interactor == "whisperx"
                if audio_mode == "transcription" and not is_whisper:
                    raise ValueError(f"Task {self.task_config.get('name', 'Transcription')} requires a Whisper model.")
                if audio_mode == "transcription" and is_whisperx:
                    raise ValueError(f"Task {self.task_config.get('name', 'Transcription')} requires a Whisper model, not WhisperX.")
                if audio_mode == "diarization" and not is_whisperx:
                    raise ValueError(f"Task {self.task_config.get('name', 'Diarization')} requires a WhisperX model.")
                if audio_mode == "qa" and (is_whisper or is_whisperx):
                    raise ValueError(f"Task {self.task_config.get('name', 'Audio QA')} does not support Whisper or WhisperX.")
                if audio_mode == "diarization":
                    params["diarize"] = True
        documents = []
        if self.task_config["RAG"]:
            documents = self.retriever.search(index_name, query, docs_k)
        llm_response = self.llm_handler(
            llm_name,
            system_prompt,
            history,
            documents,
            task_name=self.task_config.get("name"),
            **params,
        )

        if llm_response is None:
            yield "", documents
            return

        if isinstance(llm_response, str):
            yield llm_response, documents
            return

        for part in llm_response:
            yield part, documents


class RemoteHandlerClient:
    def __init__(self, endpoint, method="POST"):
        self.endpoint = endpoint
        self.method = method
    
    def __call__(self, payload):
        call_f = requests.get if self.method == "GET" else requests.post
        response = call_f(url=self.endpoint, json=payload)
        return response


class RemoteTaskHandler:
    def __init__(self, task_config):
        self.task_config = task_config
        endpoint = task_config["service"].split("-")[1]
        self.client = RemoteHandlerClient(endpoint, method="POST")
    
    def _construct_payload(self, **params):
        return {k: v for k, v in params.items()}
    
    def __call__(self, llm_name, system_prompt, history, query, docs_k, index_name, **params):
        payload = self._construct_payload(history=history, llm_name=llm_name, query=query)
        print(payload)
        response = self.client(payload)
        if response.status_code == 200:
            response = response.json()
            yield response["text"], response["documents"]
        else:
            yield "Error processing response from the remote service.", []


class Pilot3TaskHandler:
    """Bridges IP's chat flow to the Pilot_3 RAG pipeline (server.py `/query`).

    IP calls the handler with its chat `history` (list of [user, assistant] pairs,
    the last assistant turn empty) plus the current `query`. Pilot_3 instead expects
    a flat `dialog_history` (alternating user/assistant, last item the user query) and
    returns {response, retrieved_documents}. We adapt both directions here so the
    Pilot_3 app itself needs no changes.

    The pipeline owns its own LLM and retriever, so IP's LLM / Vector Store / index
    selections are ignored. `llm_name` is only used to pick the Pilot_3 `llm_type`.

    Service string format: "pilot3-<base_url>", e.g. "pilot3-http://127.0.0.1:8000".
    Optional task-config key `retriever_type` ("baseline" | "finetuned", default
    "baseline").
    """

    def __init__(self, task_config):
        self.endpoint = task_config["service"].split("pilot3-", 1)[1].rstrip("/")
        self.retriever_type = task_config.get("retriever_type", "baseline")

    @staticmethod
    def _strip_html(text):
        # IP stores rendered HTML (markdown + doc links) in assistant turns; the
        # pipeline wants plain text for the conversation history.
        return re.sub(r"<[^>]+>", "", text or "").strip()

    def _to_dialog_history(self, history):
        flat = []
        for turn in history:
            user = turn[0] if len(turn) > 0 else None
            assistant = turn[1] if len(turn) > 1 else None
            if user:
                flat.append(self._strip_html(user))
            if assistant:
                flat.append(self._strip_html(assistant))
        return flat

    def __call__(self, llm_name, system_prompt, history, query, docs_k, index_name, **params):
        dialog_history = self._to_dialog_history(history)
        if not dialog_history:
            dialog_history = [query]
        llm_type = "krikri" if "krikri" in (llm_name or "").lower() else "salamandra"
        # UI-selected retriever (passed via params) overrides the task-config default.
        retriever_type = params.get("retriever_type") or self.retriever_type
        payload = {
            "dialog_history": dialog_history,
            "retriever_type": retriever_type,
            "llm_type": llm_type,
        }
        # Forward the user-controlled generation params from IP's "LLM Parameters" panel.
        for key in ("temperature", "top_p", "max_tokens"):
            if params.get(key) is not None:
                payload[key] = params[key]
        try:
            response = requests.post(f"{self.endpoint}/query", json=payload, timeout=300)
        except requests.exceptions.RequestException as e:
            yield f"Error: could not reach the Pilot_3 pipeline at {self.endpoint} ({e}).", []
            return
        if response.status_code != 200:
            yield f"Error from Pilot_3 pipeline (HTTP {response.status_code}).", []
            return
        data = response.json()
        documents = data.get("retrieved_documents", {}).get("documents", [[]])
        documents = documents[0] if documents else []
        yield data.get("response", ""), documents
