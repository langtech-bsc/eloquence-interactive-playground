import logging

from settings import settings
from gradio_app.backend.ChatGptInteractor import ChatGptInteractor
from gradio_app.backend.HuggingfaceGenerator import HuggingfaceGenerator
from gradio_app.backend.BSCInteract import OlmoInteractor, EurollmInteractor, QwenInteractor, SalamandraInteractor, QwenInteractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self) -> None:
        self.known_llms = ["GPT-3.5"] + [llm.model for llm in settings.AVAILABLE_LLMS.values()]
        self._cache = {}
    
    def __call__(self, llm_name, system_prompt, history, documents, **params):
        llm = self._cache.get(llm_name, None)
        audio = None
        if "audio" in params and params["audio"] is not None:
            audio = params["audio"]
        if llm is None:
            llm = LLMHandler.get_llm_generator(llm_name)
            self._cache[llm_name] = llm
        llm.set_params(**params)
        try:
            response = llm(documents, history, llm_name, system_prompt, audio)
            return response
        except:
            raise RuntimeError
        
    @staticmethod
    def get_llm_generator(model_name):
        if "gpt" in model_name.lower():
            cgi = ChatGptInteractor(
                model_name=model_name, max_tokens=512, temperature=0, stream=True
            )
            return cgi
        elif model_name in ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-Instruct-v0.1"]:
            hfg = HuggingfaceGenerator(
                model_name=model_name, temperature=0, max_new_tokens=512,
            )
            return hfg
        elif model_name in settings.AVAILABLE_LLMS:
            if "olmo" in  model_name.lower():
                model_entry = settings.AVAILABLE_LLMS[model_name]
                cgi = OlmoInteractor(
                    api_endpoint=model_entry.endpoint, model_name=model_entry.model
                )
                return cgi
            elif "euro" in  model_name.lower():
                model_entry = settings.AVAILABLE_LLMS[model_name]
                cgi = EurollmInteractor(
                    api_endpoint=model_entry.endpoint, model_name=model_entry.model
                )
                return cgi
            if "salamandra" in  model_name.lower():
                model_entry = settings.AVAILABLE_LLMS[model_name]
                cgi = SalamandraInteractor(
                    api_endpoint=model_entry.endpoint, model_name=model_entry.model
                )
                return cgi
            if "qwen" in  model_name.lower():
                model_entry = settings.AVAILABLE_LLMS[model_name]
                cgi = QwenInteractor(
                    api_endpoint=model_entry.endpoint, model_name=model_entry.model
                )
                return cgi
            else:
                raise ValueError('Unknown LLM name')

        raise ValueError('Unknown LLM name')
