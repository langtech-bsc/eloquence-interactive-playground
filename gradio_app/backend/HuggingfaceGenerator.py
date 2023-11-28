import logging

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

with open('data/hftoken.txt') as f:
    HF_TOKEN = f.read().strip()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# noinspection PyTypeChecker
class HuggingfaceGenerator:
    def __init__(
            self, model_name,
            temperature: float = 0.9, max_new_tokens: int = 512,
            top_p: float = None, repetition_penalty: float = None,
            stream: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_client = InferenceClient(model_name, token=HF_TOKEN)
        self.stream = stream

        self.generate_kwargs = {
            'temperature': max(temperature, 0.1),
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'do_sample': True,
            'seed': 42,
        }

    def generate(self, messages):
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        logger.info(f'Start HuggingFace generation, model {self.hf_client.model} ...')
        stream = self.hf_client.text_generation(
            formatted_prompt, **self.generate_kwargs,
            stream=self.stream, details=True, return_full_text=not self.stream
        )

        for response in stream:
            yield response.token.text
