import requests

from gradio_app.helpers import check_llm_interface
from gradio_app.backend.rag_with_wiki import (
    retrieve_chunks,
    answer_with_wikipedia,
)

def get_task_handler(config, llm, retriver):
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
        retrieval_query = params.pop("retrieval_query", None) or query
        llm_name = self.task_config.get("llm_name", llm_name)

        if not check_llm_interface(llm_name, self.task_config["interface"], available_llms=self.llm_handler.available_llms):
            raise ValueError(f"LLM {llm_name} does not support the required interface {self.task_config['interface']}.")
        if self.task_config.get("interface") == "audio":
            audio_mode = self.task_config.get("audio_mode")
            if audio_mode:
                entry = self.llm_handler.available_llms.get(llm_name, {})
                interactor = str(entry.get("interactor", "")).strip().lower()

                is_whisper = interactor == "whisper"
                is_meusli = interactor == "meusli"
                is_whisperx = interactor == "whisperx"
                if audio_mode == "transcription" and (not is_whisper and not is_meusli):
                    raise ValueError(f"Task {self.task_config.get('name', 'Transcription')} requires a Whisper/Meusli model.")
                if audio_mode == "transcription" and is_whisperx:
                    raise ValueError(f"Task {self.task_config.get('name', 'Transcription')} requires a Whisper model, not WhisperX.")
                if audio_mode == "diarization" and not is_whisperx:
                    raise ValueError(f"Task {self.task_config.get('name', 'Diarization')} requires a WhisperX model.")
                if audio_mode == "qa" and (is_whisper or is_whisperx):
                    raise ValueError(f"Task {self.task_config.get('name', 'Audio QA')} does not support Whisper or WhisperX.")
                if audio_mode == "diarization":
                    params["diarize"] = True

        documents = []
        documents_metadata = []

        task_name = self.task_config.get("name")
        if task_name == "Wiki" and params.get("audio") is not None:
            raise ValueError("Wiki currently accepts text input.")

        if task_name == "RAG_with_Wiki":
            if not self.task_config.get("RAG"):
                raise ValueError("RAG_with_Wiki requires retrieval enabled.")

            if not check_llm_interface(
                llm_name,
                "text",
                available_llms=self.llm_handler.available_llms,
            ):
                raise ValueError(f"{task_name} requires a text model.")

            if params.get("audio") is not None:
                raise ValueError(f"{task_name} currently accepts text input.")

            chunks = retrieve_chunks(
                retriever=self.retriever,
                index_name=index_name,
                query=retrieval_query,
                top_k=docs_k,
            )

            documents = [chunk["text"] for chunk in chunks]
            documents_metadata = [chunk["metadata"] for chunk in chunks]

            answer = answer_with_wikipedia(
                model_config=self.llm_handler.available_llms[llm_name],
                system_prompt=system_prompt,
                history=history,
                documents=documents,
                task_config=self.task_config,
                **params,
            )
            yield answer, documents, documents_metadata
            return

        elif self.task_config["RAG"]:
            if hasattr(self.retriever, "search_with_metadata"):
                documents, documents_metadata = (
                    self.retriever.search_with_metadata(
                        index_name,
                        retrieval_query,
                        docs_k,
                    )
                )
            else:
                documents = self.retriever.search(
                    index_name,
                    retrieval_query,
                    docs_k,
                )
        # Wiki uses the configured agent endpoint through the normal model client.
        llm_response = self.llm_handler(
            llm_name,
            system_prompt,
            history,
            documents,
            task_name=self.task_config.get("name"),
            user_input=query,
            **params,
        )

        if llm_response is None:
            yield "", documents, documents_metadata
            return

        if isinstance(llm_response, str):
            yield llm_response, documents, documents_metadata
            return

        if isinstance(llm_response, dict):
            yield llm_response, documents, documents_metadata
            return

        for part in llm_response:
            yield part, documents, documents_metadata


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
