import logging
import os
import time

import tiktoken
import openai
import tenacity
from jinja2 import Environment, FileSystemLoader

from settings import settings
from gradio_app.helpers import reverse_doc_links

env = Environment(loader=FileSystemLoader('gradio_app/templates'))
context_template = env.get_template('context_template.j2')

OPENAI_KEY = None
key_file = 'data/openaikey.txt'
if os.path.exists(key_file):
    with open(key_file) as f:
        OPENAI_KEY = f.read().strip()

if OPENAI_KEY is None:
    OPENAI_KEY = os.getenv('OPENAI_KEY')

openai.api_key = OPENAI_KEY


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apx_num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """
    Return the number of tokens used by a list of messages.
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.info("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # logger.info()("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return apx_num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # logger.info()("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return apx_num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value, disallowed_special=()))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


class ChatGptInteractor:
    def __init__(self, model_name='gpt-3.5-turbo', max_tokens=None, temperature=None, top_p=None, stream=False):
        self.model_name = model_name
        self.generate_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        self.stream = stream
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.api_endpoint = "https://api.openai.com/v1"
        self.client = openai.OpenAI(
            api_key=OPENAI_KEY,
            base_url=self.api_endpoint
        )

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
            messages.append({
                "role": "system",
                "content": system_text
            })
        messages.append({
            "role": "user",
            "content": user_text
        })
        return messages
    
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

    def chat_completion(self, messages):
        logger.info(f'Sending request to Openai model {self.model_name} stream={self.stream} ...')
        t1 = time.time()
        try:
            completion = self._request(messages)
        except:
            logger.error("Failed generating response.")
            return ""

        if self.stream:
            return self._generator(completion)

        t2 = time.time()
        usage = completion.usage
        logger.info(
            f'Received response: {usage.prompt_tokens} in + {usage.completion_tokens} out'
            f' = {usage.total_tokens} total tokens. Time: {t2 - t1:3.1f} seconds'
        )
        return completion.choices[0].message.content

    @staticmethod
    def get_stream_text(stream_part):
        return stream_part.choices[0].delta.get("content", "")

    @staticmethod
    def _generator(completion):
        for part in completion:
            yield ChatGptInteractor.get_stream_text(part)

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
            messages=messages,
            model=self.model_name,
            stream=False,
            **self.generate_kwargs
        )
        return completion
    
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = [
            {
                "role": "system",
                "content": system_prompt + "\n" + context,
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


if __name__ == '__main__':
    cgi = ChatGptInteractor()

    for txt in [
        "Hello World!",
        "Hello",
        " World!",
        "World!",
        "World",
        "!",
        " ",
        "  ",
        "   ",
        "    ",
        "\n",
        "\n\t",
    ]:
        print(f'`{txt}` | {cgi.tokenizer.encode(txt)}')

    st = 'You are a helpful assistant and an experienced programmer, ' \
         'answering questions exactly in two rhymed sentences'
    ut = 'Explain the principle of recursion in programming'
    print('Count tokens:', cgi.count_tokens_simple(user_text=ut, system_text=st))

    print(cgi.chat_completion_simple(user_text=ut, system_text=st))
    print('---')

    cgi = ChatGptInteractor()
    for part in cgi.chat_completion_simple(user_text=ut, system_text=st):
        print(part, end='')
    print('\n---')

