import torch
import openai
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts):
        pass


class HfEmbedder(Embedder):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts):
        encoded = self.model.encode(texts, normalize_embeddings=True)
        return [list(vec) for vec in encoded]


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, texts):
        responses = openai.Embedding.create(input=texts, engine=self.model_name)
        return [response['embedding'] for response in responses['data']]


class EmbedderFactory:
    @staticmethod
    def get_embedder(type):
        if type == "sentence-transformers/all-MiniLM-L6-v2":
            return HfEmbedder(type)
        elif type == "text-embedding-ada-002":
            return OpenAIEmbedder(type)
        else:
            raise ValueError(f"Unsupported embedder type: {type}")


