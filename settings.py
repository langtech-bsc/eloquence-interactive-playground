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
    PERSISTENT_DATA_ROOT: str = os.environ.get("PERSISTENT_DATA", "/playground-data")
    LANCEDB_DIRECTORY: str = f"{PERSISTENT_DATA_ROOT}/lancedb"
    LANCEDB_TABLE_NAME: str = "table"
    VECTOR_COLUMN_NAME: str = "embedding"
    TEXT_COLUMN_NAME: str = "text"
    METADATA: str = "metadata"
    UPLOAD_DIR: str = "/tmp/uploads"
    TOP_K_RERANK: int = 5
    SUPPORTED_FILE_TYPES: list = ["pdf", "docx", "csv", "tsv", "html", "md", "txt"]
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
        "Gemma (TID)": 8192,
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
    }
    #submit_clear_row {
        gap: 12px;
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
    """
    JS_CODE: str = """
async () => {
    let mediaRecorder = null;
    let socket = null;
    let isStreaming = false;
    let streamFinalizeUrl = null;
    let recordedChunks = [];
    let recordedBlob = null;

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
            if (!isWhisperSelected(selectedAudioModel)) {
                status.innerText = "Selected model is not Whisper. Please pick a Whisper model before recording.";
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
                status.innerText = "Recording stopped. Click Submit to transcribe.";
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

    // Initialize auto-scroll
    globalThis.Scrolldown();
}
"""

settings = Settings()
