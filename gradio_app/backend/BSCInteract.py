import logging
import os
import time

import tiktoken
import openai

from gradio_app.backend.ChatGptInteractor import apx_num_tokens_from_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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
        # self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    def chat_completion_simple(
            self,
            *,
            user_text,
            system_text=None,
    ):
        return self.chat_completion(
            self._construct_messages_simple(user_text, system_text),
        )

    def count_tokens_simple(self, *, user_text, system_text=None):
        return self.count_tokens(self._construct_messages_simple(user_text, system_text))

    @staticmethod
    def _construct_messages_simple(user_text, system_text=None):
        messages = []
        if system_text is not None:
            # messages.append({
            #    "role": "system",
            #    "content": system_text
            #})
            user_text = system_text + " " + user_text
        messages.append({
            "role": "user",
            "content": user_text
        })
        return messages
    
    def __call__(self, messages):
        return self.chat_completion(messages)

    def chat_completion(self, messages):
        logger.info(f'Sending request to {self.model_name} stream={self.stream} ...')
        t1 = time.time()
        completion = self._request(messages)

        if self.stream:
            return self._generator(completion)

        t2 = time.time()
        usage = completion['usage']
        logger.info(
            f'Received response: {usage["prompt_tokens"]} in + {usage["completion_tokens"]} out'
            f' = {usage["total_tokens"]} total tokens. Time: {t2 - t1:3.1f} seconds'
        )
        return completion.choices[0].message['content']

    @staticmethod
    def get_stream_text(stream_part):
        return stream_part['choices'][0]['delta'].get('content', '')

    @staticmethod
    def _generator(completion):
        for part in completion:
            yield BSCInteractor.get_stream_text(part)

    def count_tokens(self, messages):
        return apx_num_tokens_from_messages(messages, self.model_name)

    def set_params(self, **params):
        self.generate_kwargs.update(params)

    def _request(self, messages):
        openai.api_base = self.api_endpoint
        logger.info(self.api_endpoint + " " + self.model_name)
        logger.info(len(messages))
        logger.info(str([m['role'] for m in messages]))
        for _ in range(5):
            try:
                completion = openai.ChatCompletion.create(
                    messages=messages,
                    model=self.model_name,
                    stream=self.stream,
                    request_timeout=100.0,
                    **self.generate_kwargs
                )
                return completion
            except (openai.error.Timeout, openai.error.ServiceUnavailableError):
                continue
        raise RuntimeError('Failed to connect to OpenAI (timeout error)')
