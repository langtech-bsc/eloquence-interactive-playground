import os
import logging

from jinja2 import Environment, FileSystemLoader

from settings import LLM_CONTEXT_LENGHTS, AVAILABLE_LLMS
from gradio_app.helpers import reverse_doc_links
from gradio_app.backend.ChatGptInteractor import apx_num_tokens_from_messages, ChatGptInteractor
from gradio_app.backend.HuggingfaceGenerator import HuggingfaceGenerator
from gradio_app.backend.BSCInteract import BSCInteractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = Environment(loader=FileSystemLoader('gradio_app/templates'))
context_template = env.get_template('context_template.j2')


class LLMHandler:
    def __init__(self) -> None:
        self.known_llms = ["GPT-3.5"] + [llm.model for llm in AVAILABLE_LLMS.values()]
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
            try:
                num_tokens = apx_num_tokens_from_messages(messages)  # todo for HF, it is approximation
            except:
                num_tokens = len(str(messages).split()) * 2
            if num_tokens + 512 < LLM_CONTEXT_LENGHTS[llm]:
                break
            documents.pop()
        messages = LLMHandler.construct_message_list(llm, system_prompt, context, history, audio)
        return messages
        
        
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
        elif model_name in AVAILABLE_LLMS:
            model_entry = AVAILABLE_LLMS[model_name]
            cgi = BSCInteractor(
                api_endpoint=model_entry.endpoint, model_name=model_entry.model
            )
            return cgi

        raise ValueError('Unknown LLM name')
    
    @staticmethod
    def construct_message_list(model_name, system_prompt, context, history, audio=None):
        if model_name in AVAILABLE_LLMS.keys():
            messages = []
            prepended = False
            for q, a in history:
                if len(a) == 0:  # the last message
                    q = system_prompt + " Context: " + context + " " + q
                if audio is None or len(a) != 0:
                    messages.append({
                        "role": "user",
                        "content": q,
                    })
                elif len(a) == 0:
                    messages.append({
                        "role": "user",
                        "content": [
                        #    {
                        #        "type": "text",
                        #        "text": q
                        #    },
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
                        "content": reverse_doc_links(a),
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
