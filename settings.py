MARKDOWN_SOURCE_DIR = "data/transformers/docs/source/en/"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LANCEDB_DIRECTORY = "data/lancedb"
LANCEDB_TABLE_NAME = "table"
VECTOR_COLUMN_NAME = "embedding"
TEXT_COLUMN_NAME = "text"
DOCUMENT_PATH_COLUMN_NAME = "document_path"
HF_LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
OPENAI_LLM_NAME = "gpt-3.5-turbo"

""" in symbols, approximate, without headers """
TEXT_CHUNK_SIZE = 1000

emb_sizes = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "thenlper/gte-large": 0
}

context_lengths = {
    "mistralai/Mistral-7B-Instruct-v0.1": 4096,
    "gpt-3.5-turbo": 4096,
}
