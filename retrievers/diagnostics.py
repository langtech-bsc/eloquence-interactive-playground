import json
import logging

logger = logging.getLogger(__name__)


def log_retrieval(*, task, index_name, query, documents, distances, threshold=None):
    """Log candidate distances and filter decisions without another retrieval."""
    if not logger.isEnabledFor(logging.INFO):
        return
    distances = distances if isinstance(distances, list) else []
    chunks = []
    for position, document in enumerate(documents):
        distance = distances[position] if position < len(distances) else None
        chunks.append({
            "chunk": position + 1,
            "distance": distance,
            "kept": threshold is None or (distance is not None and distance <= threshold),
            "preview": " ".join(str(document).split())[:200],
        })
    logger.info("%s retrieval: %s", task, json.dumps({
        "index": index_name,
        "query": query,
        "threshold": threshold,
        "kept": sum(chunk["kept"] for chunk in chunks),
        "retrieved": len(chunks),
        "chunks": chunks,
    }, ensure_ascii=False))
