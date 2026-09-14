import os
import numpy as np
import torch
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
FINETUNED_MODEL_ID = os.environ.get("FINETUNED_MODEL_ID", "Cutting3dg3/LaBSE-TID")
CHROMA_DB_REPO_ID = os.environ.get("CHROMA_DB_REPO_ID", "Cutting3dg3/labse-tid-chromadb")
CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    snapshot_download(repo_id=CHROMA_DB_REPO_ID, repo_type="dataset"),
)
COLLECTION_NAME = "propositions_VS"

_device = "cuda" if torch.cuda.is_available() else "cpu"


class Retriever:
    def __init__(
        self,
        n_results: int = 10,
        use_finetuned: bool = True,
        max_distance: float = 0.60,
    ):
        self.n_results = n_results
        self.use_finetuned = use_finetuned
        self.max_distance = max_distance

        model_id = FINETUNED_MODEL_ID if use_finetuned else LABSE_MODEL_NAME
        self.model = SentenceTransformer(model_id).to(_device)
        self.model.tokenizer.truncation_side = "left"

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=LABSE_MODEL_NAME
        )
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef,
        )

    def generate_embedding(self, text: list[str]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            features = self.model.tokenize(text)
            features = batch_to_device(features, _device)
            embedding = self.model(features)["sentence_embedding"]
        return embedding.detach().cpu().squeeze().numpy()

    def retrieve(self, dialog_history: list[str], n_results: int = None) -> dict:
        text = [" [SEP] ".join(dialog_history)]

        tag = "finetuned" if self.use_finetuned else "baseline"
        print(f"\n[RETRIEVER:{tag}] Input query:\n  {text[0]}")

        embedding = self.generate_embedding(text)
        raw = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results or self.n_results,
            include=["documents", "distances"],
        )

        raw_docs = raw.get("documents", [[]])[0]
        raw_dists = raw.get("distances", [[]])[0]

        docs, dists = [], []
        seen = set()
        for doc, dist in zip(raw_docs, raw_dists):
            if dist > self.max_distance:
                continue
            if doc in seen:
                continue
            seen.add(doc)
            docs.append(doc)
            dists.append(dist)

        print(
            f"\n[RETRIEVER:{tag}] Kept {len(docs)} of {len(raw_docs)} documents "
            f"(distance<={self.max_distance}, deduplicated):"
        )
        for i, (doc, dist) in enumerate(zip(docs, dists)):
            print(f"  [{i + 1}] distance={dist:.4f} | {doc}")

        return {"documents": [docs], "distances": [dists]}
