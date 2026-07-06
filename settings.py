import os
from typing import List, Optional, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

USER_FEEDBACK_FILE = "user_feedback.json"
USER_HISTORY_FILE = "history.json"
USER_PROMPTS_FILE = "prompts.json"
USER_RETRIEVERS_FILE = "retrievers.json"


def normalize_path_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ""
    prefix = prefix.strip()
    if not prefix or prefix == "/":
        return ""
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    prefix = prefix.rstrip("/")
    return prefix


class LLMEntry:

    def __init__(self, llm_entry):
        llm_entry = llm_entry.split(",")
        self.endpoint = llm_entry[0]
        self.model = llm_entry[1]
        self.name = llm_entry[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    PERSISTENT_DATA_ROOT: str = os.environ.get("PERSISTENT_DATA", "/playground-data")
    LANCEDB_DIRECTORY: str = f"{PERSISTENT_DATA_ROOT}/lancedb"
    LANCEDB_TABLE_NAME: str = "table"
    VECTOR_COLUMN_NAME: str = "embedding"
    TEXT_COLUMN_NAME: str = "text"
    METADATA: str = "metadata"
    UPLOAD_DIR: str = "/tmp/uploads"
    TOP_K_RERANK: int = 5
    SUPPORTED_FILE_TYPES: list = ["pdf", "docx", "csv", "tsv", "html", "md", "txt", "jsonl"]
    RETRIEVER_ENDPOINT: str = "http://127.0.0.1:7997"
    BASIC_CONFIG: dict = {"interface": "text", "RAG": False, "service": "local"}
    BASIC_AUDIO_CONFIG: dict = {"interface": "audio", "RAG": False, "service": "local"}

    EMBEDDING_SIZES: dict = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
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
        "Salamandra (TID)": 8196,
        "Salamandra (HF)": 8196,
        "Gemma 3 (TID)": 8192,
        "Gemma 4 (TID)": 8192,
        "Apertus (TID)": 65536,
        "sentence-transformers/all-MiniLM-L6-v2": 128,
        "thenlper/gte-large": 512,
        "text-embedding-ada-002": 1000,  # actual context length is 8191, but it's too much
    }

    INDEX_CONFIG_PATH: str = f"{PERSISTENT_DATA_ROOT}/configurations/indexes.json"
    PROMPTS_PATH: str = f"{PERSISTENT_DATA_ROOT}/configurations/prompts.json"
    TASK_CONFIG_DIR: str = f"{PERSISTENT_DATA_ROOT}/configurations/task_configs/"
    RETRIEVER_CONFIG_PATH: str = f"{PERSISTENT_DATA_ROOT}/configurations/retrievers.json"
    MODELS_PATH: str = f"{PERSISTENT_DATA_ROOT}/configurations/models.json"
    USER_WORKSPACES: str = f"{PERSISTENT_DATA_ROOT}/workspaces"
    GENERIC_UPLOAD: str = f"/tmp"
    SQL_DB: str = f"{PERSISTENT_DATA_ROOT}/ip.db"
    ROOT_PATH: str = ""
    PATH_PREFIXES: List[str] = ["/dev"]
    PATH_PREFIX_HEADERS: Tuple[str, ...] = ("x-forwarded-prefix", "x-script-name")

    @field_validator("ROOT_PATH", mode="before")
    def _normalize_root_path(cls, value):
        return normalize_path_prefix(value)

    @field_validator("PATH_PREFIXES", mode="before")
    def _normalize_path_prefixes(cls, value):
        raw_prefixes = []
        if isinstance(value, str):
            raw_prefixes = value.split(",")
        else:
            raw_prefixes = value or []

        normalized_prefixes = []
        for entry in raw_prefixes:
            normalized = normalize_path_prefix(entry)
            if normalized:
                normalized_prefixes.append(normalized)
        return normalized_prefixes

    @field_validator("PATH_PREFIX_HEADERS", mode="before")
    def _normalize_prefix_headers(cls, value):
        raw_headers = []
        if isinstance(value, str):
            raw_headers = value.split(",")
        else:
            raw_headers = value or []

        headers = []
        for header in raw_headers:
            if not header:
                continue
            cleaned = header.strip().lower()
            if cleaned:
                headers.append(cleaned)
        return tuple(headers)

    # Dialog Manager Settings
    DM_ENDPOINT: str = os.environ.get("DM_ENDPOINT", "http://127.0.0.1:8003")
    DM_SESSIONS_PATH: str = f"{PERSISTENT_DATA_ROOT}/dialog_sessions.json"
    DM_MODELS_DIR: str = f"{PERSISTENT_DATA_ROOT}/models/dialog"

    # Ollama settings for dialog manager
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11435")
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4:12b")

    # GLiNER model settings
    GLINER_MODEL_NAME: str = os.environ.get("GLINER_MODEL_NAME", "gliner2_finetuned_uns")
    GLINER_THRESHOLD: float = float(os.environ.get("GLINER_THRESHOLD", "0.5"))

    # Dialog settings
    MAX_DIALOG_TURNS: int = 10

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
    #summary_controls_row {
        align-items: center;
    }
    #summary_button_column {
        align-self: center;
    }
    #summarize_btn button {
        min-height: 56px;
        line-height: 1.2;
        white-space: pre-line;
    }
    .svelte-1mhtq7j {
        background: #565553 !important;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .gradio-container .radio label,
    .gradio-container .radio label > span {
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
    }
    .gradio-container .gradio-radio label,
    .gradio-container .gradio-radio label > span,
    .gradio-container .gradio-radio label > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
    }
    #history_panel {
        position: fixed;
        top: 8%;
        left: 50%;
        transform: translateX(-50%);
        width: min(720px, 92vw);
        max-height: 80vh;
        overflow: auto;
        background: #ffffff;
        border: 1px solid #e2e2e2;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.18);
        z-index: 1000;
    }
    #prompt_panel {
        position: fixed;
        top: 8%;
        left: 50%;
        transform: translateX(-50%);
        width: min(720px, 92vw);
        max-height: 80vh;
        overflow: auto;
        background: #ffffff;
        border: 1px solid #e2e2e2;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.18);
        z-index: 1000;
    }
    #prompt_radio {
        border: 1px solid #e2e2e2;
        border-radius: 6px;
        padding: 6px;
        background: #fafafa;
        box-sizing: border-box;
    }
    #prompt_radio label,
    #prompt_radio label > span,
    #prompt_radio label > div {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    #prompt_panel textarea {
        max-height: 220px;
        overflow-y: auto !important;
    }
    #history_radio {
        border: 1px solid #e2e2e2;
        border-radius: 6px;
        padding: 6px;
        background: #fafafa;
        box-sizing: border-box;
    }
    #history_radio label,
    #history_radio label > span,
    #history_radio label > div {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    #history_panel textarea {
        max-height: 260px;
        overflow-y: auto !important;
    }
    #history_radio .wrap,
    #prompt_radio .wrap {
        max-height: 200px;
        overflow-y: auto;
        display: block;
        padding-bottom: 12px;
        box-sizing: border-box;
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
        background: #f3f4f6 !important;
        color: #6b7280 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px;
    }
    .gallery button:hover {
        background: #e5e7eb !important;
        color: #4b5563 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px;
    }
    input[type=number] {
        width: 70px;
    }
    div.svelte-sa48pu>.form>* {
        min-width: 70px;
    }
    /* Softer bubble corners */
    #chatbot :not(.component-wrap).flex-wrap.user {
        border-radius: 22px !important;
        border-bottom-right-radius: 8px !important;
    }
    #chatbot :not(.component-wrap).flex-wrap.bot {
        border-radius: 22px !important;
        border-bottom-left-radius: 8px !important;
    }
    /* Hide thumbs up/down only for user messages */
    #chatbot .message-buttons-right {
        display: none !important;
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
    .audio-controls {
        display: flex;
        flex-direction: column;
        gap: 8px;
        width: 100%;
    }
    .audio-record-buttons {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        align-items: stretch;
    }
    #input_controls_row {
        column-gap: 12px;
        position: sticky;
        bottom: 0;
        z-index: 20;
        background: #ffffff;
        padding-top: 8px;
        padding-bottom: 8px;
        border-top: 1px solid #e5e7eb;
        flex-wrap: nowrap !important;
        overflow: hidden;
    }
    #submit_clear_row {
        gap: 8px;
    }
    /* Keep Submit/Clear buttons column at fixed width */
    #submit_clear_row {
        flex-shrink: 0;
        flex-wrap: nowrap !important;
        align-items: stretch;
    }
    #input_controls_row > div:last-child {
        flex: 0 0 auto !important;
        max-width: none !important;
        min-width: auto !important;
    }
    /* Text input takes remaining space */
    #input_controls_row > div:first-child {
        flex: 1 1 0% !important;
        min-width: 0 !important;
    }
    #submit_btn,
    #submit_btn button {
        font-weight: 700;
        min-height: 44px;
        width: 100px;
        min-width: 100px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
    }
    #clear_btn,
    #clear_btn button {
        background: #f27618 !important;
        color: #ffffff !important;
        border: 1px solid #d25610 !important;
        min-height: 44px;
        width: 100px;
        min-width: 100px;
        box-shadow: none !important;
    }
    #clear_btn:hover,
    #clear_btn button:hover,
    #clear_btn:focus-visible,
    #clear_btn button:focus-visible {
        background: #d25610 !important;
        color: #ffffff !important;
        border: 1px solid #b94c0e !important;
    }
    @media (max-width: 1366px) {
        #input_controls_row {
            padding-top: 6px;
            padding-bottom: 6px;
        }
        #submit_clear_row {
            gap: 8px;
        }
        #submit_btn,
        #submit_btn button,
        #clear_btn,
        #clear_btn button {
            min-height: 40px;
            padding-top: 8px;
            padding-bottom: 8px;
        }
    }
    @media (max-width: 1100px) {
        #input_controls_row {
            column-gap: 8px;
            padding-top: 4px;
            padding-bottom: 4px;
        }
        #submit_clear_row {
            gap: 6px;
        }
        #submit_btn,
        #submit_btn button,
        #clear_btn,
        #clear_btn button {
            min-height: 38px;
            padding-top: 6px;
            padding-bottom: 6px;
            font-size: 14px;
        }
    }
    .audio-record-buttons button {
        width: 100%;
        height: 44px;
        line-height: 44px;
        padding: 0 16px;
    }
    .audio-playback audio {
        width: 100%;
        margin-top: 2px;
    }
    #recordstatus {
        background: #f5f5f5 !important;
        color: #333 !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 6px 8px;
        max-width: 100%;
        font-size: 12px;
        border: 1px solid #e2e2e2;
        font-family: "Lucida Console", "Courier New", monospace;
    }
    .gradio-container footer {
        display: none !important;
    }
    #custom_project_footer {
        margin-top: 12px;
        padding: 10px 12px;
        text-align: center;
        font-size: 14px;
        color: #4b5563;
        border-top: 1px solid #e5e7eb;
    }
    #custom_project_footer a {
        color: #018f69;
        font-weight: 600;
        text-decoration: none;
    }
    #custom_project_footer a:hover {
        text-decoration: underline;
    }
    /* Toggle button for config panel */
    #toggle_config_btn {
        width: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
        height: 36px;
        padding: 0;
        border-radius: 50%;
        font-size: 20px;
        line-height: 36px;
        text-align: center;
        background: #d1d5db !important;
        color: #374151 !important;
        border: none !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
        cursor: pointer;
        flex-shrink: 0;
    }
    #toggle_config_btn:hover {
        background: #b0b5bc !important;
    }
    /* Config panel header row */
    #config_panel_header {
        display: none !important;
    }
    #config_panel {
        border: none;
        border-radius: 10px;
        padding: 0;
        background: transparent;
        transition: flex 0.3s ease, min-width 0.3s ease;
    }
    """
    JS_CODE: str = """
async () => {
    let mediaRecorder = null;
    let socket = null;
    let isStreaming = false;
    let streamFinalizeUrl = null;
    let recordedChunks = [];
    let recordedBlob = null;
    let focusRetryTimer = null;
    let postReplyFocusTimer = null;
    let placeholderCleared = false;

    const getSelectedRadioValue = (containerId) => {
        const container = document.getElementById(containerId);
        if (!container) return null;
        const checked = container.querySelector('input[type="radio"]:checked');
        return checked ? checked.value : null;
    };

    const getTaskConfig = () => {
        const selected = document.querySelector('#task_config input[type="radio"]:checked');
        if (!selected) return null;
        try {
            return JSON.parse(selected.value);
        } catch (e) {
            return null;
        }
    };

    const isWhisperSelected = (modelName) => {
        if (!modelName) return false;
        return String(modelName).toLowerCase().includes("whisper");
    };

    const isWhisperXSelected = (modelName) => {
        if (!modelName) return false;
        return String(modelName).toLowerCase().includes("whisperx");
    };

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
        if (isStreaming) {
            return;
        }
        const taskConfig = getTaskConfig();
        if (!taskConfig || taskConfig.interface !== "audio") {
            status.innerText = "Please select an audio task before recording.";
            return;
        }
        const audioMode = taskConfig.audio_mode;
        const audioQaMode = getSelectedRadioValue("audio_qa_mode");
        if (audioMode === "qa" && !audioQaMode) {
            status.innerText = "Please select an Audio QA mode before recording.";
            return;
        }

        const selectedAudioModel = getSelectedRadioValue("llm_name");
        if (audioMode === "transcription" || (audioMode === "qa" && audioQaMode === "whisper_llm")) {
            if (!selectedAudioModel) {
                status.innerText = "Please select a Whisper model before recording.";
                return;
            }
            if (!isWhisperSelected(selectedAudioModel) || isWhisperXSelected(selectedAudioModel)) {
                status.innerText = "Selected model is not Whisper. Please pick a Whisper model before recording.";
                return;
            }
        }

        if (audioMode === "diarization") {
            if (!selectedAudioModel) {
                status.innerText = "Please select the WhisperX model before recording.";
                return;
            }
            if (!isWhisperXSelected(selectedAudioModel)) {
                status.innerText = "Selected model is not WhisperX. Please pick WhisperX before recording.";
                return;
            }
        }

        if (audioMode === "qa" && audioQaMode === "speech_llm") {
            if (!selectedAudioModel) {
                status.innerText = "Please select a Speech LLM model before recording.";
                return;
            }
            if (isWhisperSelected(selectedAudioModel)) {
                status.innerText = "Selected model is Whisper. Please pick a Speech LLM model before recording.";
                return;
            }
        }

        if (audioMode === "qa" && audioQaMode === "whisper_llm") {
            const selectedTextModel = getSelectedRadioValue("text_llm_name");
            if (!selectedTextModel) {
                status.innerText = "Please select a Text LLM before recording.";
                return;
            }
        }

        if (!window.isSecureContext) {
            status.innerText = "Recording requires HTTPS (or localhost).";
            return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            return;
        }

        if (!streamFinalizeUrl) {
            streamFinalizeUrl = new URL("/stream_finalize", window.location.origin).toString();
        }

        const handleStream = (stream) => {
            mediaRecorder = new MediaRecorder(stream);
            isStreaming = true;
            recordedChunks = [];
            recordedBlob = null;
            status.innerText = "Microphone recording... streaming audio.";

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            mediaRecorder.onstop = () => {
                const audioEl = document.getElementById('recorded_audio');
                if (recordedChunks.length === 0) {
                    status.innerText = "No audio captured.";
                    return;
                }
                const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
                recordedBlob = blob;
                status.innerText = "Recording stopped. Click Submit to process the audio.";
                if (audioEl) {
                    audioEl.src = URL.createObjectURL(blob);
                }
            };
            mediaRecorder.start(250); // Send data every 250ms
        };

        const handleError = (err) => {
            console.error("getUserMedia error:", err);
            status.innerText = "Error: " + (err && err.name ? err.name : "getUserMedia");
        };

        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(handleStream)
            .catch(handleError);
    };

    // Stop streaming audio
    globalThis.stopStreaming = function() {
        const status = document.getElementById('recordstatus');
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.requestData();
            mediaRecorder.stop();
            isStreaming = false;
            if (mediaRecorder.stream) {
                mediaRecorder.stream.getTracks().forEach((track) => track.stop());
            }
        } else {
            status.innerText = "Recording is not active.";
        }
    };

    const isAudioTaskSelected = () => {
        const selected = document.querySelector('#task_config input[type="radio"]:checked');
        if (!selected) return false;
        try {
            const config = JSON.parse(selected.value);
            return config && config.interface === "audio";
        } catch (e) {
            return false;
        }
    };

    const getInputTextarea = () => {
        return document.querySelector('#input_textbox textarea') ||
               document.querySelector('#input_controls_row textarea');
    };

    const clearPlaceholderOnFirstMessage = () => {
        if (placeholderCleared) return;
        const textarea = getInputTextarea();
        if (!textarea) return;
        const hasText = !!(textarea.value && textarea.value.trim().length > 0);
        if (hasText) {
            textarea.placeholder = "";
            placeholderCleared = true;
        }
    };

    const isVisible = (el) => {
        if (!el) return false;
        return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
    };

    const queueInputFocus = () => {
        if (focusRetryTimer) {
            clearInterval(focusRetryTimer);
            focusRetryTimer = null;
        }
        let attempts = 0;
        focusRetryTimer = setInterval(() => {
            attempts += 1;
            const textarea = getInputTextarea();
            if (textarea && !textarea.disabled && isVisible(textarea)) {
                textarea.focus();
                const textLen = textarea.value ? textarea.value.length : 0;
                textarea.setSelectionRange(textLen, textLen);
                clearInterval(focusRetryTimer);
                focusRetryTimer = null;
            } else if (attempts > 60) {
                clearInterval(focusRetryTimer);
                focusRetryTimer = null;
            }
        }, 100);
    };

    const uploadRecordedAudio = async () => {
        if (!recordedBlob) return null;
        if (!streamFinalizeUrl) {
            streamFinalizeUrl = new URL("/stream_finalize", window.location.origin).toString();
        }
        const formData = new FormData();
        formData.append("audio_file", recordedBlob, "recording.webm");
        const resp = await fetch(streamFinalizeUrl, {
            method: "POST",
            body: formData
        });
        if (!resp.ok) {
            throw new Error(`Finalize upload failed: ${resp.status}`);
        }
        const json = await resp.json();
        recordedBlob = null;
        return json;
    };

    const submitBtn = document.getElementById("submit_btn");
    if (submitBtn) {
        submitBtn.addEventListener("click", async (event) => {
            clearPlaceholderOnFirstMessage();
            queueInputFocus();
            if (!isAudioTaskSelected()) {
                return;
            }
            const status = document.getElementById('recordstatus');
            if (!recordedBlob) {
                status.innerText = "No audio recorded. Please record audio first.";
                event.preventDefault();
                event.stopImmediatePropagation();
                return;
            }
            event.preventDefault();
            event.stopImmediatePropagation();
            try {
                status.innerText = "Uploading audio...";
                const json = await uploadRecordedAudio();
                if (json) {
                    status.innerText = `Upload complete (${json.size_received} bytes). Submitting...`;
                }
                const btn = document.getElementById('trigger_audio_submit');
                if (btn) btn.click();
            } catch (err) {
                console.error("Finalize upload error:", err);
                status.innerText = "Error uploading final recording.";
            }
        }, true);
    }

    const bindEnterFocusHandler = () => {
        const textarea = getInputTextarea();
        if (!textarea || textarea.dataset.focusBound === "1") return;
        textarea.dataset.focusBound = "1";
        textarea.addEventListener("input", () => {
            clearPlaceholderOnFirstMessage();
        });
        textarea.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                clearPlaceholderOnFirstMessage();
                queueInputFocus();
            }
        });
    };

    bindEnterFocusHandler();

    // Add tooltip to Clear button
    const clearBtn = document.querySelector('#clear_btn button') || document.getElementById('clear_btn');
    if (clearBtn) {
        clearBtn.title = "Clears the entire conversation history and resets the system prompt";
    }

    const inputObserver = new MutationObserver(() => {
        bindEnterFocusHandler();
    });
    inputObserver.observe(document.body, { childList: true, subtree: true });

    const chatNode = document.querySelector('[aria-label="chatbot conversation"]');
    if (chatNode) {
        const replyObserver = new MutationObserver(() => {
            if (postReplyFocusTimer) {
                clearTimeout(postReplyFocusTimer);
            }
            // Debounce during token streaming and refocus once updates settle.
            postReplyFocusTimer = setTimeout(() => {
                queueInputFocus();
            }, 250);
        });
        replyObserver.observe(chatNode, { childList: true, subtree: true, characterData: true });
    }

    // Initialize auto-scroll
    globalThis.Scrolldown();
}
"""


settings = Settings()
