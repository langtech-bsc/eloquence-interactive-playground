import time

import tiktoken
import openai


with open('data/openaikey.txt') as f:
    OPENAI_KEY = f.read().strip()
openai.api_key = OPENAI_KEY


def num_tokens_from_messages(messages, model):
    """
    Return the number of tokens used by a list of messages.
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
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
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
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
    def __init__(self, model_name='gpt-3.5-turbo'):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    def chat_completion_simple(
            self,
            *,
            user_text,
            system_text=None,
            max_tokens=None,
            temperature=None,
            stream=False,
    ):
        return self.chat_completion(
            self._construct_messages_simple(user_text, system_text),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
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

    def chat_completion(
            self,
            messages,
            max_tokens=None,
            temperature=None,
            stream=False,
    ):
        print(f'Sending request to {self.model_name} stream={stream} ...')
        t1 = time.time()
        completion = self._request(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        if stream:
            return completion
        t2 = time.time()
        usage = completion['usage']
        print(
            f'Received response: {usage["prompt_tokens"]} in + {usage["completion_tokens"]} out'
            f' = {usage["total_tokens"]} total tokens. Time: {t2 - t1:3.1f} seconds'
        )
        return completion.choices[0].message['content']

    @staticmethod
    def get_stream_text(stream_part):
        return stream_part['choices'][0]['delta'].get('content', '')

    def count_tokens(self, messages):
        return num_tokens_from_messages(messages, self.model_name)

    def _request(self, *args, **kwargs):
        for _ in range(5):
            try:
                completion = openai.ChatCompletion.create(
                    *args, **kwargs,
                    request_timeout=100.0,
                )
                return completion
            except (openai.error.Timeout, openai.error.ServiceUnavailableError):
                continue
        raise RuntimeError('Failed to connect to OpenAI (timeout error)')


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

    for part in cgi.chat_completion_simple(user_text=ut, system_text=st, stream=True):
        print(cgi.get_stream_text(part), end='')
    print('\n---')

