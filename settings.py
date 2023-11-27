MARKDOWN_SOURCE_DIR = "data/transformers/docs/source/en/"
LANCEDB_DIRECTORY = "data/lancedb"
LANCEDB_TABLE_NAME = "table"
VECTOR_COLUMN_NAME = "embedding"
TEXT_COLUMN_NAME = "text"
DOCUMENT_PATH_COLUMN_NAME = "document_path"

# LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
LLM_NAME = "gpt-3.5-turbo"
# EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_NAME = "text-embedding-ada-002"


emb_sizes = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "thenlper/gte-large": 1024,
    "text-embedding-ada-002": 1536,
}

context_lengths = {
    "mistralai/Mistral-7B-Instruct-v0.1": 4096,
    "gpt-3.5-turbo": 4096,
    "sentence-transformers/all-MiniLM-L6-v2": 128,
    "thenlper/gte-large": 512,
    "text-embedding-ada-002": 8191,
}
