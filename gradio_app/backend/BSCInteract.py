import logging
import time
import re

import openai
import tenacity
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.ChatGptInteractor import apx_num_tokens_from_messages
from gradio_app.helpers import reverse_doc_links, encode_audio_stream
from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = Environment(loader=FileSystemLoader("gradio_app/templates"))
context_template = env.get_template("context_template.j2")


# Base class for BSC-based interactors, providing common functionality for constructing messages, handling requests, and formatting responses.
# Specific model interactors (Olmo, Eurollm, Qwen, etc.) inherit from this class and implement their own message construction logic based on the expected input format of the respective models.
class BSCInteractor:
    def __init__(
        self, api_endpoint, model_name, api_key=None, max_tokens=None, temperature=None, top_p=None, stream=False
    ):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.generate_kwargs = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        logger.info("Creating with endpoint and name:" + api_endpoint + model_name)
        self.stream = stream
        self.client = openai.OpenAI(base_url=self.api_endpoint, api_key=api_key)
        logger.info("Creating with endpoint and name:" + api_endpoint + model_name)

    def __call__(self, documents, history, llm, system_prompt, audio=None):
        messages = self.build_messages(documents, history, llm, system_prompt, audio)
        return self.chat_completion(messages)

    def build_messages(self, documents, history, llm, system_prompt, audio):
        context = ""
        if len(documents) > 0:
            while len(documents) > 0:
                context = context_template.render(documents=documents)
                messages = self._construct_message_list(llm, system_prompt, context, history, audio)
                try:
                    num_tokens = apx_num_tokens_from_messages(messages)  # todo for HF, it is approximation
                except:
                    num_tokens = len(str(messages).split()) * 2
                max_ctx = settings.LLM_CONTEXT_LENGHTS.get(llm, 4096)  # default fallback to 4096
                if num_tokens + 512 < max_ctx:
                    break
                documents.pop()
        messages = self._construct_message_list(llm, system_prompt, context, history, audio)
        return messages

    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        raise NotImplementedError

    def chat_completion(self, messages):
        logger.info(f"Sending request to {self.model_name} stream={self.stream} ...")
        t1 = time.time()
        try:
            logger.info(f"Sent messages: {messages}")
            completion = self._request(messages)
            print("Received completion:", completion)
        except Exception as exc:
            logger.exception("Failed generating response!")
            raise RuntimeError(self._format_request_error(exc))

        if self.stream:
            return self._generator(completion)

        t2 = time.time()
        usage = completion.usage
        logger.info(
            f"Received response: {usage.prompt_tokens} in + {usage.completion_tokens} out"
            f" = {usage.total_tokens} total tokens. Time: {t2 - t1:3.1f} seconds"
        )
        return completion.choices[0].message.content

    @staticmethod
    def get_stream_text(stream_part):
        return stream_part.choices[0].delta.get("content", "")

    @staticmethod
    def _generator(completion):
        for part in completion:
            yield BSCInteractor.get_stream_text(part)

    @staticmethod
    def _extract_content_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)
        return ""

    @classmethod
    def _extract_answer_text(cls, completion):
        if not getattr(completion, "choices", None):
            return ""
        message = getattr(completion.choices[0], "message", None)
        if message is None:
            return ""

        if isinstance(message, dict):
            return cls._extract_content_text(message.get("content"))

        content = getattr(message, "content", "")
        return cls._extract_content_text(content)

    def count_tokens(self, messages):
        return apx_num_tokens_from_messages(messages, self.model_name)

    def set_params(self, **params):
        self.generate_kwargs.update(params)

    def _format_request_error(self, exc):
        """Provide actionable error messages for transport/retry failures."""
        root_exc = exc
        if isinstance(exc, tenacity.RetryError):
            last_attempt = getattr(exc, "last_attempt", None)
            if last_attempt is not None:
                try:
                    last_exc = last_attempt.exception()
                    if last_exc is not None:
                        root_exc = last_exc
                except Exception:
                    pass

        err_name = type(root_exc).__name__
        err_text = str(root_exc).strip() or repr(root_exc)
        return f"LLM request failed for model '{self.model_name}' at '{self.api_endpoint}': " f"{err_name}: {err_text}"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
    def _request(self, messages, override_kwargs=None):
        logger.info(self.api_endpoint + " " + self.model_name)
        logger.info(len(messages))
        logger.info(str([m["role"] for m in messages]))
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=self.stream, **self.generate_kwargs
        )

        return completion


class OlmoInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                if context:
                    q = system_prompt + " Context: " + context + " " + q
                else:
                    q = system_prompt + " " + q
            if len(a) != 0:
                messages.append(
                    {
                        "role": "user",
                        "content": q,
                    }
                )
            elif len(a) == 0:
                messages.append({"role": "user", "content": [{"type": "text", "text": q}]})
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
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
                messages.append(
                    {
                        "role": "system",
                        "content": context,
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": q,
                }
            )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class EurollmInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                if context:
                    q = system_prompt + " Context: " + context + " " + q
                else:
                    q = system_prompt + " " + q
            if len(a) != 0:
                messages.append(
                    {
                        "role": "user",
                        "content": q,
                    }
                )
            elif len(a) == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": q},
                        ],
                    }
                )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class QwenInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                if context:
                    q = system_prompt + " Context: " + context + " " + q
                else:
                    q = system_prompt + " " + q
            if audio is None or len(a) != 0:
                messages.append(
                    {
                        "role": "user",
                        "content": q,
                    }
                )
            elif len(a) == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            #    {
                            #      "type": "text",
                            #       "text": q
                            #   },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": encode_audio_stream(audio), "format": "wav"},
                            }
                        ],
                    }
                )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class GemmaInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for idx, (q, a) in enumerate(history):
            if idx == 0 and system_prompt:
                q = system_prompt + "\n\n" + q
            if len(a) == 0 and context:
                q = "Context: " + context + "\n\n" + q
            messages.append(
                {
                    "role": "user",
                    "content": q,
                }
            )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class ApertusInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        system_content = system_prompt or ""
        if context:
            if system_content:
                system_content = system_content + " Context: " + context
            else:
                system_content = "Context: " + context
        if system_content:
            messages.append(
                {
                    "role": "system",
                    "content": system_content,
                }
            )
        for q, a in history:
            messages.append(
                {
                    "role": "user",
                    "content": q,
                }
            )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class SDialogInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        system_content = (system_prompt or "").strip()
        if context:
            if system_content:
                system_content = system_content + " Context: " + context
            else:
                system_content = "Context: " + context
        if system_content:
            messages.append(
                {
                    "role": "system",
                    "content": system_content,
                }
            )
        for q, a in history:
            messages.append(
                {
                    "role": "user",
                    "content": q,
                }
            )
            if len(a) != 0:
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class SalamandraInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        if context:
            system_content = system_prompt + " Context: " + context
        else:
            system_content = system_prompt
        messages = [
            {
                "role": "system",
                "content": system_content,
            }
        ]
        for q, a in history:
            messages.append(
                {
                    "role": "user",
                    "content": q,
                }
            )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class WhisperInteractor(BSCInteractor):
    def __init__(
        self,
        api_endpoint,
        model_name,
        api_key=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
        stream=False,
        transcription_kwargs=None,
    ):
        super().__init__(api_endpoint, model_name, api_key, max_tokens, temperature, top_p, stream)
        self.transcription_kwargs = transcription_kwargs or {}

    def __call__(self, documents, history, llm, system_prompt, audio, language=None):
        from io import BytesIO

        audio = BytesIO(audio)
        audio.name = "in.wav"
        transcription = self.client.audio.transcriptions.create(
            file=audio, model=self.model_name, language="es", temperature=0.0
        )
        return transcription.text
