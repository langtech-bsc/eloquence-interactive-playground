MARKDOWN_SOURCE_DIR = "data/transformers/docs/source/en/"
LANCEDB_DIRECTORY = "data/lancedb"
LANCEDB_TABLE_NAME = "table"
VECTOR_COLUMN_NAME = "embedding"
TEXT_COLUMN_NAME = "text"
DOCUMENT_PATH_COLUMN_NAME = "document_path"

CHUNK_POLICY = "md"
# CHUNK_POLICY = "txt"

EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# EMBED_NAME = "text-embedding-ada-002"

TOP_K_RANK = 50
TOP_K_RERANK = 5

emb_sizes = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "thenlper/gte-large": 1024,
    "text-embedding-ada-002": 1536,
}

thresh_distances = {
    "sentence-transformers/all-MiniLM-L6-v2": 1.2,
    "text-embedding-ada-002": 0.5,
}

context_lengths = {
    "mistralai/Mistral-7B-Instruct-v0.1": 4096,
    "GeneZC/MiniChat-3B": 4096,
    "gpt-3.5-turbo": 4096,
    "sentence-transformers/all-MiniLM-L6-v2": 128,
    "thenlper/gte-large": 512,
    "text-embedding-ada-002": 1000,  # actual context length is 8191, but it's too much
}
