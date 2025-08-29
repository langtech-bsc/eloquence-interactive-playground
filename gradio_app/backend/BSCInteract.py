import logging
import time

import openai
import tenacity
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.ChatGptInteractor import apx_num_tokens_from_messages
from gradio_app.helpers import reverse_doc_links, encode_audio_stream
from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = Environment(loader=FileSystemLoader('gradio_app/templates'))
context_template = env.get_template('context_template.j2')


class BSCInteractor:
    def __init__(self, api_endpoint, model_name, max_tokens=None, temperature=None, top_p=None, stream=False):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.generate_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        self.stream = stream
        self.client = openai.OpenAI(
            base_url=self.api_endpoint,
            api_key="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"
        )
        logger.info("Creating with endpoint and name:" + api_endpoint + model_name)
    
    def __call__(self, documents, history, llm, system_prompt, audio=None):
        messages = self.build_messages(documents, history, llm, system_prompt, audio)
        return self.chat_completion(messages)
    
    def build_messages(self, documents, history, llm, system_prompt, audio):
        context = ""
        while len(documents) > 0:
            context = context_template.render(documents=documents)
            messages = self._construct_message_list(llm, system_prompt, context, history, audio)
            try:
                num_tokens = apx_num_tokens_from_messages(messages)  # todo for HF, it is approximation
            except:
                num_tokens = len(str(messages).split()) * 2
            if num_tokens + 512 < settings.LLM_CONTEXT_LENGHTS[llm]:
                break
            documents.pop()
        messages = self._construct_message_list(llm, system_prompt, context, history, audio)
        return messages
    
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        raise NotImplementedError

    def chat_completion(self, messages):
        logger.info(f'Sending request to {self.model_name} stream={self.stream} ...')
        t1 = time.time()
        try:
            completion = self._request(messages)
        except:
            logger.error("Failed generating response!")
            return ""

        if self.stream:
            return self._generator(completion)

        t2 = time.time()
        if "usage" in completion:
            usage = completion.usage
        else:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        logger.info(
            f'Received response: {usage["prompt_tokens"]} in + {usage["completion_tokens"]} out'
            f' = {usage["total_tokens"]} total tokens. Time: {t2 - t1:3.1f} seconds'
        )
        return completion.choices[0].message.content

    @staticmethod
    def get_stream_text(stream_part):
        return stream_part.choices[0].delta.get('content', '')

    @staticmethod
    def _generator(completion):
        for part in completion:
            yield BSCInteractor.get_stream_text(part)

    def count_tokens(self, messages):
        return apx_num_tokens_from_messages(messages, self.model_name)

    def set_params(self, **params):
        self.generate_kwargs.update(params)

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
    def _request(self, messages):
        logger.info(self.api_endpoint + " " + self.model_name)
        logger.info(len(messages))
        logger.info(str([m['role'] for m in messages]))
        completion = self.client.chat.completions.create(
            model="tgi",
            messages=messages,
            stream=self.stream,
            **self.generate_kwargs
        )

        return completion


class OlmoInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                q = system_prompt + " Context: " + context + " " + q
            if len(a) != 0:
                messages.append({
                    "role": "user",
                    "content": q,
                })
            elif len(a) == 0:
                messages.append({
                    "role": "user",
                    "content": [
                       {
                           "type": "text",
                           "text": q
                       }
                    ]

                })
            if len(a) != 0:  # some of the previous LLM answers
                messages.append({
                    "role": "assistant",
                    "content": reverse_doc_links(a),
                })
        return messages


class SalamandraInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
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
                    "content": reverse_doc_links(a),
                })
        return messages


class EurollmInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                q = system_prompt + " Context: " + context + " " + q
            if len(a) != 0:
                messages.append({
                    "role": "user",
                    "content": q,
                })
            elif len(a) == 0:
                messages.append({
                    "role": "user",
                    "content": [
                       {
                           "type": "text",
                           "text": q
                       },
                    ]

                })
            if len(a) != 0:  # some of the previous LLM answers
                messages.append({
                    "role": "assistant",
                    "content": reverse_doc_links(a),
                })
        return messages


class QwenInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
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
                      #      "type": "text",
                     #       "text": q
                     #   },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encode_audio_stream(audio),
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
        return messages


class SalamandraInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = [
            {
                "role": "system",
                "content": system_prompt + " Context: " + context,
            }
        ]
        for q, a in history:
            messages.append({
                "role": "user",
                "content": q,
            })
            if len(a) != 0:  # some of the previous LLM answers
                messages.append({
                    "role": "assistant",
                    "content": reverse_doc_links(a),
                })
        return messages


class WhisperInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                q = system_prompt + " Context: " + context + " " + q
            if len(a) != 0:
                messages.append({
                    "role": "user",
                    "content": q,
                })
            elif len(a) == 0:
                messages.append({
                    "role": "user",
                    "content": [
                       {
                           "type": "text",
                           "text": q
                       },
                    ]

                })
            if len(a) != 0:  # some of the previous LLM answers
                messages.append({
                    "role": "assistant",
                    "content": reverse_doc_links(a),
                })
        return messages
