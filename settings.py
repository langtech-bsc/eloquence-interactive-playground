MARKDOWN_SOURCE_DIR = "data/transformers/docs/source/en/"
DATA_ROOT = "./app"
LANCEDB_DIRECTORY = f"{DATA_ROOT}/eloquence-playground/lancedb"
LANCEDB_TABLE_NAME = "table"
VECTOR_COLUMN_NAME = "embedding"
TEXT_COLUMN_NAME = "text"
DOCUMENT_PATH_COLUMN_NAME = "document_path"

TOP_K_RANK = 50
TOP_K_RERANK = 5
SUPPORTED_FILE_TYPES = ["pdf", "docx", "csv", "tsv", "html", "md", "txt"]

EMBEDDING_SIZES = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    # "text-embedding-ada-002": 1536,
}

THRESHOLD_DISTANCES = {
    "sentence-transformers/all-MiniLM-L6-v2": 1.2,
    "text-embedding-ada-002": 0.5,
}

LLM_CONTEXT_LENGHTS = {
    "mistralai/Mistral-7B-Instruct-v0.1": 4096,
    "tiiuae/falcon-180B-chat": 2048,
    "meta-llama/Meta-Llama-3-8B": 2048,
    "bsc": 2048,
    "bsc2": 2048,
    "bsc3": 2048,
    "GeneZC/MiniChat-3B": 4096,
    "gpt-3.5-turbo": 4096,
    "sentence-transformers/all-MiniLM-L6-v2": 128,
    "thenlper/gte-large": 512,
    "text-embedding-ada-002": 1000,  # actual context length is 8191, but it's too much
}

INDEX_CONFIG_PATH = "configurations/indexes.json"
PROMPTS_PATH = "configurations/prompts.json"
TASK_CONFIG_DIR = "configurations/task_configs/"
USER_WORKSPACES = f"{DATA_ROOT}/eloquence-playground/workspaces"
GENERIC_UPLOAD = f"uploads"
SQL_DB = "ip.db"

CSS = """
button.secondary {
    background: #018f69;
    border-radius: 6px;
    max-height:4em;

}
button.secondary:hover {
    background: #016f49;
    border-radius: 6px;
    max-height:4em;
}
.svelte-1mhtq7j {
    background: #565553 !important;
    color: white;
}
.svelte-1mhtq7j:hover, .svelte-1mhtq7j:hover > *, .svelte-1mhtq7j.selected, .svelte-1mhtq7j.selected > * {
    background:  #f2d518 !important;
     color: #363533;
}

.svelte-1mhtq7j.selected {
    border: 3px double #363533;
}

label.selected {
    background: #f2d518!;
    text: black;
}
.gallery button {
    background: #f27618;
    border-radius: 6px;
}
.gallery button:hover {
    background: #d25610;
    border-radius: 6px;
}
input[type=number] {
    width: 70px;
}
div.svelte-sa48pu>.form>* {
    min-width: 70px;
}
.svelte-1mhtq7j {
    background: #f2d518;
}
#status, #status textarea {
    font-weight: bold !important;
    color: white !important;
    background: #f27618 !important;
    border-radius: 6px;
}
.description {
    color: #999999;
    font-family: "Lucida Console", "Courier New", monospace;
}
"""
