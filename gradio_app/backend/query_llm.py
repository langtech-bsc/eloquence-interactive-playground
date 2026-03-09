import logging
from copy import deepcopy

from settings import settings
from gradio_app.backend.ChatGptInteractor import ChatGptInteractor
from gradio_app.backend.HuggingfaceGenerator import HuggingfaceGenerator
from gradio_app.backend.BSCInteract import OlmoInteractor, EurollmInteractor, QwenInteractor, SalamandraInteractor, GemmaInteractor, ApertusInteractor, WhisperInteractor, SDialogInteractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
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
        
    def get_llm_generator(self, model_name, task_name=None):
        model_entry = self.available_llms[model_name]
        if task_name == "SDialog":
            cgi = SDialogInteractor(
                api_endpoint=model_entry["api_endpoint"], model_name=model_entry["model_name"], api_key=model_entry.get("api_key", None)
            )
            return cgi
        if "gpt" in model_name.lower():
            cgi = ChatGptInteractor(
                model_name=model_entry["model_name"], max_tokens=512, temperature=0, stream=False, api_endpoint=model_entry["api_endpoint"], api_key=model_entry.get("api_key", None)
            )
            return cgi
        elif model_name in ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-Instruct-v0.1"]:
            hfg = HuggingfaceGenerator(
                model_name=model_entry["model_name"], temperature=0, max_new_tokens=512, api_endpoint=model_entry["api_endpoint"], api_key=model_entry.get("api_key", None)
            )
            return hfg
        elif model_name in self.available_llms.keys():
            if "olmo" in  model_name.lower():
                cgi = OlmoInteractor(
                    api_endpoint=model_entry["api_endpoint"], model_name=model_entry["model_name"], api_key=model_entry.get("api_key", None)
                )
                return cgi
            if "euro" in  model_name.lower():
                cgi = EurollmInteractor(
                    api_endpoint=model_entry["api_endpoint"], model_name=model_entry["model_name"], api_key=model_entry.get("api_key", None)
                )
                return cgi
            if "salamandra" in  model_name.lower():
                cgi = SalamandraInteractor(
                    api_endpoint=model_entry["api_endpoint"], model_name=model_entry["model_name"], api_key=model_entry.get("api_key", None)
                )
                return cgi
            if "qwen" in  model_name.lower():
                cgi = QwenInteractor(
                    api_endpoint=model_entry["api_endpoint"], model_name=model_entry["model_name"], api_key=model_entry.get("api_key", None)
                )
                return cgi
            if "gemma" in  model_name.lower():
                cgi = GemmaInteractor(
                    api_endpoint=model_entry["api_endpoint"], model_name=model_entry["model_name"], api_key=model_entry.get("api_key", None)
                )
                return cgi
            if "apertus" in  model_name.lower():
                cgi = ApertusInteractor(
                    api_endpoint=model_entry["api_endpoint"], model_name=model_entry["model_name"], api_key=model_entry.get("api_key", None)
                )
                return cgi
            if "whisper" in  model_name.lower():
                cgi = WhisperInteractor(
                    api_endpoint=model_entry["api_endpoint"], model_name=model_entry["model_name"], api_key=model_entry.get("api_key", None)
                )
                return cgi
            else:
                raise ValueError('Unknown LLM name')

        raise ValueError('Unknown LLM name')
