import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

USER_FEEDBACK_FILE = "user_feedback.json"
USER_HISTORY_FILE = "history.json"
USER_PROMPTS_FILE = "prompts.json"
USER_RETRIEVERS_FILE = "retrievers.json"

class LLMEntry:

    def __init__(self, llm_entry):
        llm_entry = llm_entry.split(",")
        self.endpoint = llm_entry[0]
        self.model = llm_entry[1]
        self.name = llm_entry[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    AVAILABLE_LLMS: dict = {}
    MARKDOWN_SOURCE_DIR: str = "data/transformers/docs/source/en/"
    PERSISTENT_DATA_ROOT: str = os.environ.get("PERSISTENT_DATA", "/app")
    LANCEDB_DIRECTORY: str = f"{PERSISTENT_DATA_ROOT}/lancedb"
    LANCEDB_TABLE_NAME: str = "table"
    VECTOR_COLUMN_NAME: str = "embedding"
    TEXT_COLUMN_NAME: str = "text"
    METADATA: str = "metadata"
    UPLOAD_DIR: str = "/tmp/uploads"
    TOP_K_RANK: int = 50
    TOP_K_RERANK: int = 5
    SUPPORTED_FILE_TYPES: list = ["pdf", "docx", "csv", "tsv", "html", "md", "txt"]
    RETRIEVER_ENDPOINT: str = "http://127.0.0.1:7999"
    BASIC_CONFIG: dict = {"interface": "text", "RAG": False, "service": "local"}
    
    EMBEDDING_SIZES: dict = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }

    THRESHOLD_DISTANCES: dict = {
        "sentence-transformers/all-MiniLM-L6-v2": 1.2,
    }

    LLM_CONTEXT_LENGHTS: dict = {
        "mistralai/Mistral-7B-Instruct-v0.1": 4096,
        "tiiuae/falcon-180B-chat": 2048,
        "meta-llama/Meta-Llama-3-8B": 2048,
        "GeneZC/MiniChat-3B": 4096,
        "gpt-3.5-turbo": 4096,
        "Qwen2-Audio": 4096,
        "EuroLLM": 4096,
        "Salamandra (MN5)": 8196,
        "Salamandra (MN5)": 8196,
        "sentence-transformers/all-MiniLM-L6-v2": 128,
        "thenlper/gte-large": 512,
        "text-embedding-ada-002": 1000,  # actual context length is 8191, but it's too much
    }

    INDEX_CONFIG_PATH: str = f"{PERSISTENT_DATA_ROOT}/configurations/indexes.json"
    PROMPTS_PATH: str = f"{PERSISTENT_DATA_ROOT}/configurations/prompts.json"
    TASK_CONFIG_DIR: str = f"{PERSISTENT_DATA_ROOT}/configurations/task_configs/"
    RETRIEVER_CONFIG_PATH: str = f"{PERSISTENT_DATA_ROOT}/configurations/retrievers.json"
    USER_WORKSPACES: str = f"{PERSISTENT_DATA_ROOT}/workspaces"
    GENERIC_UPLOAD: str = f"uploads"
    SQL_DB: str = f"{PERSISTENT_DATA_ROOT}/ip.db"

    @field_validator("AVAILABLE_LLMS", mode="before")
    def parse_available_llms(cls, v):
        # os.environ.get("OPENAI_API_ENDPOINT_URLs", "[]")
        avail = [LLMEntry(entry) for entry in v]
        return {entry.name: entry for entry in avail}


    CSS: str = """
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
    #ingestion_status, #ingestion_status textarea {
        background: #f27618;
        color: #ffffff;
        padding: 2px
        border-radius: 6px;
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
    #recordstatus {
        background: rgb(242, 213, 24) !important;
        color: black !important;
        font-weight: bold;
        border-radius: 4px;
        padding: 5px;
        max-width: 300px;
        font-family: "Lucida Console", "Courier New", monospace;
    }
    """
    JS_CODE: str = """
async () => {
    let mediaRecorder = null;
    let socket = null;
    let isStreaming = false;

    // Auto-scroll the chatbot window
    globalThis.Scrolldown = function() {
        const targetNode = document.querySelector('[aria-label="chatbot conversation"]');
        if (!targetNode) return;

        const config = { attributes: true, childList: true, subtree: true };
        const callback = (mutationList, observer) => {
            targetNode.scrollTop = targetNode.scrollHeight;
        };
        const observer = new MutationObserver(callback);
        observer.observe(targetNode, config);
    };

    // Start streaming audio from the microphone
    globalThis.startStreaming = function() {
        const status = document.getElementById('recordstatus');
        status.innerText = "Requesting microphone access...";

        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                isStreaming = true;
                status.innerText = "Microphone recording... streaming audio.";

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && isStreaming) {
                        const formData = new FormData();
                        formData.append("audio_chunk", event.data);

                        fetch("https://eloquence.lt.bsc.es/stream", {
                            method: "POST",
                            body: formData
                        }).catch(err => {
                            console.error("Audio upload error:", err);
                            status.innerText = "Error during streaming.";
                        });
                    }
                };
                mediaRecorder.start(250); // Send data every 250ms
            })
            .catch(err => {
                console.error("getUserMedia error:", err);
                status.innerText = "Error: " + err.name;
            });
    };

    // Stop streaming audio
    globalThis.stopStreaming = function() {
        const status = document.getElementById('recordstatus');
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
            isStreaming = false;
            status.innerText = "Recording stopped.";
        }
    };

    // Initialize auto-scroll
    globalThis.Scrolldown();
}
"""

settings = Settings()
