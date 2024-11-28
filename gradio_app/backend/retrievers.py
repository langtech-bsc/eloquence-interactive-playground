from gradio_app.backend.embedders import EmbedderFactory
from settings import VECTOR_COLUMN_NAME, TEXT_COLUMN_NAME, THRESHOLD_DISTANCES


class LanceDBRetriever:
    def __init__(self, db, threshold) -> None:
        self.emb_cache = {}
        self.threshold = threshold
        self.db = db
    
    def __call__(self, index_name, query, embedding_type):
        if embedding_type in self.emb_cache:
            embedder = self.emb_cache[embedding_type]
        else:
            embedder = EmbedderFactory.get_embedder(embedding_type)
            self.emb_cache[embedding_type] = embedder

        table = self.db.open_table(index_name)
        query_vec = embedder.embed([query])[0]
        documents = table.search(query_vec, vector_column_name=VECTOR_COLUMN_NAME)
        if self.threshold:
            thresh_dist = THRESHOLD_DISTANCES[embedding_type]
            thresh_dist = max(thresh_dist, min(d['_distance'] for d in documents))
            documents = [d for d in documents if d['_distance'] <= self.threshold]
        documents = [doc[TEXT_COLUMN_NAME] for doc in documents]
        return documents
