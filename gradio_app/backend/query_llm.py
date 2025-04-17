import os
import logging

from jinja2 import Environment, FileSystemLoader

from settings import LLM_CONTEXT_LENGHTS
from gradio_app.backend.ChatGptInteractor import apx_num_tokens_from_messages, ChatGptInteractor
from gradio_app.backend.HuggingfaceGenerator import HuggingfaceGenerator
from gradio_app.backend.BSCInteract import BSCInteractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = Environment(loader=FileSystemLoader('gradio_app/templates'))
context_template = env.get_template('context_template.j2')


class LLMHandler:
    def __init__(self) -> None:
        self.known_llms = ["gpt-3.5-turbo", "meta-llama/Meta-Llama-3-8B", "bsc", "bsc2", "bsc3"]
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
        messages = LLMHandler.build_messages(documents, history, llm_name, system_prompt, audio)
        try:
            response = llm(messages)
            return response
        except:
            raise RuntimeError


    @staticmethod
    def build_messages(documents, history, llm, system_prompt, audio=None):
        context = ""
        while len(documents) > 0:
            context = context_template.render(documents=documents)
            messages = LLMHandler.construct_message_list(llm, system_prompt, context, history, audio)
            num_tokens = apx_num_tokens_from_messages(messages)  # todo for HF, it is approximation
            if num_tokens + 512 < LLM_CONTEXT_LENGHTS[llm]:
                break
            documents.pop()
        messages = LLMHandler.construct_message_list(llm, system_prompt, context, history, audio)
        return messages
        
        
    @staticmethod
    def get_llm_generator(llm_name):
        if llm_name == "gpt-3.5-turbo":
            cgi = ChatGptInteractor(
                model_name=llm_name, max_tokens=512, temperature=0, stream=True
            )
            return cgi
        if llm_name in ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-Instruct-v0.1"]:
            hfg = HuggingfaceGenerator(
                model_name=llm_name, temperature=0, max_new_tokens=512,
            )
            return hfg
        if llm_name.lower() in ["bsc", "bsc2", "bsc3"]:
            if llm_name == "bsc":
                bsc_api_endpoint = os.getenv("OPENAI_API_ENDPOINT_URL")
            elif llm_name == "bsc2":
                bsc_api_endpoint = os.getenv("OPENAI_API_ENDPOINT_URL_2")
            else:
                bsc_api_endpoint = os.getenv("OPENAI_API_ENDPOINT_URL_3")
            logger.info(bsc_api_endpoint)
            cgi = BSCInteractor(
                api_endpoint=bsc_api_endpoint, model_name="llama3-8B"
            )
            return cgi

        raise ValueError('Unknown LLM name')
    
    @staticmethod
    def construct_message_list(llm_name, system_prompt, context, history, audio=None):
        if "bsc" in llm_name:
            messages = []
            prepended = False
            for q, a in history:
                if len(a) == 0:  # the last message
                    q = system_prompt + " Context: " + context + " " + q
                if audio is None:
                    messages.append({
                        "role": "user",
                        "content": q,
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": q
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio,
                                    "format": "wav"
                                }
                            }
                        ]

                    })
                if len(a) != 0:  # some of the previous LLM answers
                    messages.append({
                        "role": "assistant",
                        "content": a,
                    })
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]
            for q, a in history:
                if len(a) == 0:  # the last message
                    messages.append({
                        "role": "system",
                        "content": context,
                    })
                messages.append({
                    "role": "user",
                    "content": q,
                })
                if len(a) != 0:  # some of the previous LLM answers
                    messages.append({
                        "role": "assistant",
                        "content": a,
                    }) 
        return messages


def get_message_constructor(llm_name, system_prompt):
    if llm_name in ['gpt-3.5-turbo']:
        return get_construct_openai_messages(system_prompt)
    if llm_name in ["meta-llama/Meta-Llama-3-8B",
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "tiiuae/falcon-180B-chat",
                    "GeneZC/MiniChat-3B",
                    ]:
        return construct_mistral_messages
    raise ValueError('Unknown LLM name')

