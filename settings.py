MARKDOWN_DIR_TO_SCRAPE = "data/transformers/docs/source/en/"
TEXT_CHUNKS_DIR = "data/docs_dump"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LANCEDB_DIRECTORY = "data/lancedb"
LANCEDB_TABLE_NAME = "table"
VECTOR_COLUMN_NAME = "embedding"
TEXT_COLUMN_NAME = "text"
HF_LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
OPENAI_LLM_NAME = "gpt-3.5-turbo"

context_lengths = {
    "mistralai/Mistral-7B-Instruct-v0.1": 4096,
    "gpt-3.5-turbo": 4096,
}
