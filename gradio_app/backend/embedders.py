from abc import ABC, abstractmethod
import os

import torch
import openai
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


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


class BGEM3Embedder(Embedder):
    """Adapter exposing BGE-M3 through the LangChain embedding interface."""

    def __init__(self, model_name):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "BGE-M3 retrieval requires the FlagEmbedding package."
            ) from exc

        configured_device = os.environ.get("BGE_M3_DEVICE")
        if configured_device:
            device = configured_device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=device.startswith("cuda"),
            devices=device,
        )

    def _encode(self, texts):
        vectors = self.model.encode(
            list(texts),
            batch_size=32,
            max_length=2048,
        )["dense_vecs"]
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

    def embed(self, texts):
        return self._encode(texts)

    def embed_documents(self, texts):
        return self._encode(texts)

    def embed_query(self, text):
        return self._encode([text])[0]


class EmbedderFactory:
    @staticmethod
    def get_embedder(model):
        if model in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]:
            return HuggingFaceEmbeddings(model_name=model)
        elif model == "BAAI/bge-m3":
            return BGEM3Embedder(model)
        else:
            raise ValueError(f"Unsupported embedding model: {model}")
