import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from settings import *


cross_encoder = None
cross_enc_tokenizer = None


@torch.no_grad()
def rerank_with_cross_encoder(cross_enc_name, documents, query):
    if cross_enc_name is None or len(documents) <= 1:
        return documents

    global cross_encoder, cross_enc_tokenizer
    if cross_encoder is None or cross_encoder.name_or_path != cross_enc_name:
        cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_enc_name)
        cross_encoder.eval()
        cross_enc_tokenizer = AutoTokenizer.from_pretrained(cross_enc_name)

    features = cross_enc_tokenizer(
        [query] * len(documents), documents, padding=True, truncation=True, return_tensors="pt"
    )
    scores = cross_encoder(**features).logits.squeeze()
    ranks = torch.argsort(scores, descending=True)
    documents = [documents[i] for i in ranks[:TOP_K_RERANK]]
    return documents




