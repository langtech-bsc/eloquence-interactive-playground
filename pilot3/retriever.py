import os
import numpy as np
import torch
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import batch_to_device

LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
FINETUNED_MODEL_ID = os.environ.get("FINETUNED_MODEL_ID", "Cutting3dg3/LaBSE-TID")
CHROMA_DB_REPO_ID = os.environ.get("CHROMA_DB_REPO_ID", "Cutting3dg3/labse-tid-chromadb")
CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    snapshot_download(repo_id=CHROMA_DB_REPO_ID, repo_type="dataset"),
)
COLLECTION_NAME = "propositions_VS"

# English all-MiniLM-L6-v2 fine-tuned with TID on doc2dial (training code: Pilot_3/mMiniLM).
# Its propositions were re-encoded by the frozen base model into a separate Chroma DB.
# tag -> (fine-tuned model, base model, (Chroma DB repo, collection))
MINILM_SPECS = {
    "minilm_l6_tid": (
        os.environ.get("MINILM_L6_MODEL_ID", "Cutting3dg3/MiniLM_L6_TID_d"),
        "sentence-transformers/all-MiniLM-L6-v2",
        (os.environ.get("MINILM_L6_CHROMA_DB_REPO_ID", "Cutting3dg3/minilm-tid-chromadb"),
         "propositions_minilm_l6"),
    ),
}
# As in training: the last 5 turns, joined by the tokenizer's [SEP], oldest tokens cut first.
MINILM_LAST_TURNS = 5
MINILM_MAX_TOKENS = 512

# BAAI/bge-m3 fine-tuned with TID on English + Greek doc2dial (Greek machine-translated).
# Its propositions were encoded by the frozen base model into a separate Chroma DB.
BGE_M3_MODEL_ID = os.environ.get("BGE_M3_MODEL_ID", "Cutting3dg3/BGE_M3_TID_d_en_el")
BGE_M3_CHROMA_DB_REPO_ID = os.environ.get("BGE_M3_CHROMA_DB_REPO_ID", "Cutting3dg3/bge-m3-tid-chromadb")
BGE_M3_COLLECTION_NAME = "propositions_bge_m3"
# BGE-M3 cosines run higher than LaBSE's: on-topic top-20 results score >= 0.65 and
# off-topic top hits 0.48-0.64, so keep similarity >= 0.60.
BGE_M3_MAX_DISTANCE = 0.40
# As in training: the last 5 turns, joined by a literal " [SEP] ", oldest tokens cut first.
BGE_M3_LAST_TURNS = 5
BGE_M3_MAX_TOKENS = 512

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_minilm(model_id: str, base_model: str) -> SentenceTransformer:
    # The checkpoint was saved by sentence-transformers 6 / transformers 5, whose
    # module paths and tokenizer config this image can't load.
    # Build the same Transformer + mean-pooling stack by hand; the tokenizer is
    # unchanged by fine-tuning, so take it from the base model. Its Normalize module
    # is left out: the index uses cosine space.
    transformer = models.Transformer(
        model_id,
        max_seq_length=MINILM_MAX_TOKENS,
        tokenizer_name_or_path=base_model,
    )
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[transformer, pooling])


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
        self.tag = "finetuned" if use_finetuned else "baseline"
        self.separator = " [SEP] "
        self.last_turns = None

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
        turns = dialog_history[-self.last_turns:] if self.last_turns else dialog_history
        text = [self.separator.join(turns)]

        tag = self.tag
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

        # Both collections use cosine space, so distance = 1 - cosine similarity.
        sims = [1.0 - d for d in dists]
        return {"documents": [docs], "distances": [dists], "similarities": [sims]}


class MiniLMRetriever(Retriever):
    """A TID-fine-tuned MiniLM (see MINILM_SPECS) over its base model's Chroma index."""

    def __init__(self, tag: str = "minilm_l6_tid", n_results: int = 10, max_distance: float = 0.60):
        self.n_results = n_results
        self.use_finetuned = True
        self.max_distance = max_distance
        self.tag = tag
        self.last_turns = MINILM_LAST_TURNS

        model_id, base_model, (db_repo_id, collection_name) = MINILM_SPECS[tag]
        self.model = _load_minilm(model_id, base_model).to(_device)
        self.model.tokenizer.truncation_side = "left"
        self.separator = f" {self.model.tokenizer.sep_token} "

        path = snapshot_download(repo_id=db_repo_id, repo_type="dataset")
        client = chromadb.PersistentClient(path=path)
        self.collection = client.get_collection(name=collection_name)


class BgeM3Retriever(Retriever):
    """BGE-M3 fine-tuned with TID (English + Greek) over its base model's Chroma index."""

    def __init__(self, n_results: int = 10, max_distance: float = BGE_M3_MAX_DISTANCE):
        self.n_results = n_results
        self.use_finetuned = True
        self.max_distance = max_distance
        self.tag = "bge_m3_tid"
        self.last_turns = BGE_M3_LAST_TURNS
        self.separator = " [SEP] "

        # Saved by sentence-transformers 6, whose module paths this image can't load:
        # rebuild its Transformer + CLS-pooling stack. Its Normalize module is left out,
        # as for MiniLM: the index uses cosine space.
        transformer = models.Transformer(BGE_M3_MODEL_ID, max_seq_length=BGE_M3_MAX_TOKENS)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="cls")
        self.model = SentenceTransformer(modules=[transformer, pooling]).to(_device)
        self.model.tokenizer.truncation_side = "left"

        path = snapshot_download(repo_id=BGE_M3_CHROMA_DB_REPO_ID, repo_type="dataset")
        client = chromadb.PersistentClient(path=path)
        self.collection = client.get_collection(name=BGE_M3_COLLECTION_NAME)
