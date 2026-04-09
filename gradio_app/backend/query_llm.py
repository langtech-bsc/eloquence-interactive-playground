import logging
from copy import deepcopy

from settings import settings
from gradio_app.backend.ChatGptInteractor import ChatGptInteractor
from gradio_app.backend.HuggingfaceGenerator import HuggingfaceGenerator
from gradio_app.backend.BSCInteract import (
    OlmoInteractor,
    EurollmInteractor,
    QwenInteractor,
    SalamandraInteractor,
    GemmaInteractor,
    ApertusInteractor,
    WhisperInteractor,
    WhisperXInteractor,
    SDialogInteractor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    INTERACTOR_CLASSES = {
        "olmo": OlmoInteractor,
        "eurollm": EurollmInteractor,
        "euro": EurollmInteractor,
        "salamandra": SalamandraInteractor,
        "qwen": QwenInteractor,
        "gemma": GemmaInteractor,
        "apertus": ApertusInteractor,
        "whisper": WhisperInteractor,
        "whisperx": WhisperXInteractor,
        "sdialog": SDialogInteractor,
    }

    def __init__(self, available_llms) -> None:
        logger.info(f"Available LLMs: {list(available_llms.keys())}")
        self.available_llms = available_llms
        self._cache = {}
    
    def __call__(self, llm_name, system_prompt, history, documents, **params):
        task_name = params.pop("task_name", None)
        cache_key = f"{task_name}::{llm_name}" if task_name else llm_name
        llm = self._cache.get(cache_key, None)
        audio = None
        language = None
        if "audio" in params:
            audio = deepcopy(params["audio"])
            del params["audio"]
        if "language" in params:
            language = params["language"]
            del params["language"]
        if llm is None:
            llm = self.get_llm_generator(llm_name, task_name=task_name)
            self._cache[cache_key] = llm
        llm.set_params(**params)
        try:
            response = llm(documents, history, llm_name, system_prompt, audio, language=language)
            return response
        except Exception as exc:
            logger.exception("LLM request failed for %s", llm_name)
            raise RuntimeError(str(exc))

    @staticmethod
    def _base_interactor_kwargs(model_entry):
        return {
            "api_endpoint": model_entry["api_endpoint"],
            "model_name": model_entry["model_name"],
            "api_key": model_entry.get("api_key"),
        }

    def _resolve_interactor(self, model_name, model_entry, task_name=None):
        if task_name == "SDialog":
            return "sdialog"

        configured = str(model_entry.get("interactor", "")).strip().lower()
        return configured
        
    def get_llm_generator(self, model_name, task_name=None):
        if model_name not in self.available_llms:
            raise ValueError(f"Unknown LLM name: {model_name}")

        model_entry = self.available_llms[model_name]
        base_kwargs = self._base_interactor_kwargs(model_entry)
        interactor = self._resolve_interactor(model_name, model_entry, task_name=task_name)

        if interactor == "chatgpt":
            return ChatGptInteractor(
                model_name=model_entry["model_name"],
                max_tokens=512,
                temperature=0,
                stream=False,
                api_endpoint=model_entry["api_endpoint"],
                api_key=model_entry.get("api_key"),
            )

        if interactor == "huggingface":
            return HuggingfaceGenerator(
                model_name=model_entry["model_name"],
                temperature=0,
                max_new_tokens=512,
                api_endpoint=model_entry["api_endpoint"],
                api_key=model_entry.get("api_key"),
            )

        interactor_cls = self.INTERACTOR_CLASSES.get(interactor)
        if interactor_cls is not None:
            return interactor_cls(**base_kwargs)

        raise ValueError(f"Unknown interactor '{interactor}' for model '{model_name}'")
