import shutil
import datetime
import os
import json
from typing import Any, Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import markdown
import gradio as gr
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.task_handlers import get_task_handler
from gradio_app.backend.query_llm import LLMHandler
from gradio_app.helpers import replace_doc_links, _load_json, _save_json, remove_html_tags, \
    extract_docs_from_rendered_template, _get_user_filepath, check_llm_interface
from retrievers.client import RetrieverClient
from settings import settings, USER_FEEDBACK_FILE, USER_HISTORY_FILE, USER_PROMPTS_FILE, USER_RETRIEVERS_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
LANGUAGE_CHOICES = [
    ("English", "en"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Greek", "el"),
    ("Serbian", "sr"),
]
dynamic_data = {
    "retriever_instance": None,
    "feedback_df": None,
    "audio_buffer": []
}
with open(settings.MODELS_PATH) as f:
    available_llms = json.load(f)
    available_llms = {model["display_name"]: model for model in available_llms}
llm_handler = LLMHandler(available_llms)

# --- Jinja2 Templates ---
env = Environment(loader=FileSystemLoader('gradio_app/templates'))
context_template = env.get_template('context_template.j2')
context_html_template = env.get_template('context_html_template.j2')


def perform_ingest(
        index_name: str,
        chunk_size: int,
        percentile: int,
        embed_name: str,
        file_paths: List[str],
        splitting_strategy: str,
        retriever_address: str,
        append: bool = False,
        snippet_metadata: Optional[Dict[str, Any]] = None,
        snippet_turns: Optional[List[Dict[str, Any]]] = None,
):
    """Handles the document ingestion process."""
    if not file_paths:
        raise gr.Error("You must upload at least one file.")
    if isinstance(file_paths, str):
        file_paths = file_paths.split(",")

    gr.Info("Ingesting documents...")
    retriever = RetrieverClient(endpoint=retriever_address)
    uploaded_files = []
    for fp in file_paths:
        src_path = fp.name if hasattr(fp, "name") else str(fp)
        src_path = src_path.strip()
        if os.path.isabs(src_path) and os.path.exists(src_path):
            uploaded_files.append(src_path)
        else:
            uploaded_files.append(os.path.join(settings.GENERIC_UPLOAD, os.path.basename(src_path)))

    try:
        retriever.create_vs(
            uploaded_files,
            chunk_size,
            percentile,
            embed_name,
            index_name,
            splitting_strategy,
            append=append,
            metadata=snippet_metadata,
            turns=snippet_turns,
        )
    finally:
        generic_root = os.path.abspath(settings.GENERIC_UPLOAD)
        for uploaded_file in uploaded_files:
            abs_uploaded = os.path.abspath(uploaded_file)
            in_generic_root = abs_uploaded == generic_root or abs_uploaded.startswith(generic_root + os.sep)
            if in_generic_root and os.path.isfile(abs_uploaded):
                try:
                    os.unlink(abs_uploaded)
                except OSError:
                    pass
    gr.Info("Ingestion successful!")


def load_task(task_config):
    task_config = json.loads(task_config)
    rag_enabled = task_config.get("name") == "RAG"
    show_summary = task_config.get("name") in {"Summarization", "DM"}
    audio_mode = task_config.get("audio_mode")
    interfaces = task_config.get("interface", "text")
    interfaces = interfaces if isinstance(interfaces, list) else [interfaces]
    if interfaces == ["audio"]:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(interactive=True),
            gr.update(
                choices=LANGUAGE_CHOICES,
                value="en",
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(visible=audio_mode == "qa", value="whisper_llm" if audio_mode == "qa" else None),
            gr.update(visible=rag_enabled),
            gr.update(visible=rag_enabled),
            gr.update(visible=rag_enabled),
            gr.update(visible=show_summary),
            gr.update(visible=show_summary),
            gr.update(visible=False),
            gr.update(visible=rag_enabled),
        )
    elif "text" in interfaces and "audio" in interfaces:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(
                choices=LANGUAGE_CHOICES,
                value="en",
                visible=False,
            ),
            gr.update(visible=True),
            gr.update(visible=False, value=None),
            gr.update(visible=rag_enabled),
            gr.update(visible=rag_enabled),
            gr.update(visible=rag_enabled),
            gr.update(visible=show_summary),
            gr.update(visible=show_summary),
            gr.update(visible=False),
            gr.update(visible=rag_enabled),
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(
                choices=LANGUAGE_CHOICES,
                value="en",
                visible=False,
            ),
            gr.update(visible=True),
            gr.update(visible=False, value=None),
            gr.update(visible=rag_enabled),
            gr.update(visible=rag_enabled),
            gr.update(visible=rag_enabled),
            gr.update(visible=show_summary),
            gr.update(visible=show_summary),
            gr.update(visible=False),
            gr.update(visible=rag_enabled),
        )


def _process_llm_request(llm_name, system_prompt, history, query, docs_k, index_name, task_config, retriever_instance,
                         **kwargs):
    """
    Core generator function to handle LLM requests. Yields response parts and documents.
    This is the central logic used by both Gradio and FastAPI.
    """
    task_handler = get_task_handler(task_config, llm_handler, retriever_instance)

    if not query and not kwargs.get("audio"):
        raise ValueError("Empty query submitted.")

    history = history + [[query, ""]]

    # Extract audio data and language if present in kwargs
    audio_data = kwargs.get("audio")
    language = kwargs.get("language")
    render_doc_links = kwargs.get("render_doc_links", True)
    metadata_sink = kwargs.get("metadata_sink")

    logger.info('Starting LLM stream and document retrieval...')
    stream = task_handler(
        llm_name,
        system_prompt,
        history,
        query,
        docs_k,
        index_name,
        temperature=kwargs.get("temperature", 1.0),
        top_p=kwargs.get("top_p", 0.95),
        max_tokens=kwargs.get("max_tokens", 300),
        audio=audio_data,
        language=language,
        retrieval_query=kwargs.get("retrieval_query"),
        session_id=kwargs.get("session_id")
    )

    for stream_item in stream:
        if len(stream_item) == 3:
            part, documents, documents_metadata = stream_item
        else:
            part, documents = stream_item
            documents_metadata = []

        if metadata_sink is not None:
            metadata_sink["documents"] = documents_metadata
        if isinstance(part, dict):
            if metadata_sink is not None:
                metadata_sink["transcription"] = part
            part = part.get("text", "")
        history[-1][1] += part
        if render_doc_links:
            history[-1][1] = replace_doc_links(history[-1][1])
        yield history, documents


# --- Ingestion Tab ---
def upload_file_for_ingest(files: List[str]) -> gr.update:
    """Copies uploaded files to a temporary location for ingestion."""
    out_files = []
    os.makedirs(settings.GENERIC_UPLOAD, exist_ok=True)
    for file_path in files:
        gr.Info(f"Uploading '{os.path.basename(file_path)}'...")
        shutil.copy(file_path.name, settings.GENERIC_UPLOAD)
        out_files.append(os.path.basename(file_path.name))
    return gr.update(value=", ".join(out_files))

def validate_ingestion_inputs(
    index_name, embedder, files, chunk_length, percentile, retriever_addr,
):
    """Validates all inputs on the Ingestion tab before starting the process."""
    if not all([index_name, embedder, files, retriever_addr]):
        raise gr.Error(
            "Please fill all required fields: Index Name, Embedder, Vector "
            "Store, and upload at least one file."
        )

    for f in files:
        if not any(f.name.endswith(suff) for suff in settings.SUPPORTED_FILE_TYPES):
            raise gr.Error(f"File '{os.path.basename(f.name)}' has an unsupported file type.")

    try:
        if not 0 < int(chunk_length) <= 2000:
            raise ValueError()
    except (TypeError, ValueError):
        raise gr.Error("'Chunk Length' must be an integer between 1 and 2000.")

    try:
        if not 0 < int(percentile) <= 100:
            raise ValueError()
    except (TypeError, ValueError):
        raise gr.Error("'Percentile' must be an integer between 1 and 100.")


# --- Playground Tab ---
def get_dynamic_components(request: gr.Request) -> tuple:
    """Loads all dynamic data (prompts, history, etc.) when the UI loads."""
    user = request.username
    feedback_df, avail_cols = _get_feedback_df()
    task_configs_radio, _ = _get_task_configs()
    retrievers_radio, retrievers = _get_retrievers(user)
    # Set default retriever instance
    if retrievers:
        dynamic_data["retriever_instance"] = RetrieverClient(endpoint=list(retrievers.values())[0])
    online_choices, _ = _get_online_models(llm_handler.available_llms)

    return (
        task_configs_radio,
        _get_prompts(user),
        _get_historical_prompts(user),
        online_choices,
        retrievers_radio,
        retrievers_radio,  # For both Playground and Ingestion tabs
        feedback_df,
        avail_cols
    )


def change_retriever(selected_retr_endpoint: str) -> gr.Radio:
    """Updates the available indexes when a different Vector Store is selected."""
    dynamic_data["retriever_instance"] = RetrieverClient(endpoint=selected_retr_endpoint)
    choices = dynamic_data["retriever_instance"].list_vs()
    return gr.Radio(label="Index name", choices=choices)


def save_system_prompt(request: gr.Request, system_prompt: str):
    """Saves a new system prompt for the current user."""
    if not system_prompt:
        gr.Warning("Cannot save an empty system prompt.")
        return gr.update()
    filepath = _get_user_filepath(request.username, USER_PROMPTS_FILE)
    prompts = _load_json(filepath)
    prompt_snippet = system_prompt.strip().replace("\n", " ")
    name = f"{datetime.date.today().isoformat()} - {prompt_snippet[:30] if prompt_snippet else 'Untitled'}"
    prompts.append({"name": name, "system_prompt": system_prompt})
    _save_json(filepath, prompts)
    gr.Info("System prompt saved successfully!")
    return gr.update(choices=_build_prompt_choices(prompts), value=None)
    # To refresh the dropdown, we would need to return a new gr.Dropdown object
    # For simplicity, user needs to reload to see the new prompt.


def store_history(request: gr.Request, history: List[List[str]], system_prompt: str):
    """Saves the current conversation history for the user."""
    filepath = _get_user_filepath(request.username, USER_HISTORY_FILE)
    logs = _load_json(filepath)
    date_str = datetime.date.today().isoformat()
    first_user_msg = ""
    if history and history[0]:
        first_user_msg = (history[0][0] or "").strip()
    prompt_snippet = (system_prompt or "").strip()
    if prompt_snippet:
        prompt_snippet = prompt_snippet.replace("\n", " ")[:30]
    else:
        prompt_snippet = (first_user_msg[:30] if first_user_msg else "Untitled")
    log_name = f"{date_str} - {prompt_snippet} ({len(history)} turns)"
    logs.append({"history": history, "system_prompt": system_prompt, "name": log_name})
    _save_json(filepath, logs)
    gr.Info(f"History saved for user '{request.username}'.")
    return gr.update(choices=_build_history_choices(logs), value=None)


def _build_history_choices(logs):
    choices = []
    for idx in range(len(logs) - 1, -1, -1):
        entry = logs[idx]
        label = entry.get("name", f"History {idx + 1}")
        choices.append((label, str(idx)))
    return choices


def _find_history_entry(logs, selected_log: str):
    if not selected_log:
        return None
    if selected_log.isdigit():
        idx = int(selected_log)
        if 0 <= idx < len(logs):
            return logs[idx]
        return None
    for entry in reversed(logs):
        name = (entry.get("name") or "").strip()
        if name == selected_log:
            return entry
    # Fallback for truncated or slightly altered labels
    for entry in reversed(logs):
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        if name.startswith(selected_log) or selected_log.startswith(name):
            return entry
    return None


def _find_prompt_entry(prompts, selected_log: str):
    if not selected_log:
        return None
    selected_log = selected_log.strip()
    if selected_log.isdigit():
        idx = int(selected_log)
        if 0 <= idx < len(prompts):
            return prompts[idx]
        return None
    for entry in reversed(prompts):
        name = (entry.get("name") or "").strip()
        if name == selected_log:
            return entry
    for entry in reversed(prompts):
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        if name.startswith(selected_log) or selected_log.startswith(name):
            return entry
    return None


def _format_history_preview(history, system_prompt: str = "", max_turns: int = 6, max_chars: int = 2000) -> str:
    if not history:
        if system_prompt:
            return f"System Prompt: {system_prompt}"
        return ""
    lines = []
    if system_prompt:
        lines.append(f"System Prompt: {system_prompt}")
        lines.append("")
    for user_msg, assistant_msg in history[:max_turns]:
        user_text = remove_html_tags(user_msg or "")
        assistant_text = remove_html_tags(assistant_msg or "")
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")
        lines.append("")
    preview = "\n".join(lines).strip()
    if len(history) > max_turns:
        preview += "\n\n…"
    return preview[:max_chars]


def load_history(request: gr.Request, selected_log: str):
    """Loads a saved conversation history for the user."""
    if not selected_log:
        gr.Warning("Please select a history entry.")
        return gr.update(value=[]), gr.update(value="")
    filepath = _get_user_filepath(request.username, USER_HISTORY_FILE)
    logs = _load_json(filepath, default=[])
    match = _find_history_entry(logs, selected_log)
    if not match:
        gr.Warning("Selected history entry not found.")
        return gr.update(value=[]), gr.update(value="")
    system_prompt = match.get("system_prompt", match.get("prompt", ""))
    return match.get("history", []), gr.update(value=system_prompt)


def load_history_preview(request: gr.Request, selected_log: str):
    """Updates the preview for the selected history without loading it."""
    if not selected_log:
        return gr.update(value="")
    filepath = _get_user_filepath(request.username, USER_HISTORY_FILE)
    logs = _load_json(filepath, default=[])
    match = _find_history_entry(logs, selected_log)
    if not match:
        return gr.update(value="")
    system_prompt = match.get("system_prompt", match.get("prompt", ""))
    preview = _format_history_preview(match.get("history", []), system_prompt=system_prompt)
    return gr.update(value=preview)


def load_history_confirm(request: gr.Request, selected_log: str):
    """Loads history after user confirmation and closes the history panel."""
    if not selected_log:
        gr.Warning("Please select a history entry.")
        return gr.update(value=[]), gr.update(value=""), gr.update(visible=False)
    filepath = _get_user_filepath(request.username, USER_HISTORY_FILE)
    logs = _load_json(filepath, default=[])
    match = _find_history_entry(logs, selected_log)
    if not match:
        gr.Warning("Selected history entry not found.")
        return gr.update(value=[]), gr.update(value=""), gr.update(visible=False)
    system_prompt = match.get("system_prompt", match.get("prompt", ""))
    return match.get("history", []), gr.update(value=system_prompt), gr.update(visible=False)


def load_system_prompt_preview(request: gr.Request, selected_log: str):
    """Updates the preview for the selected system prompt without applying it."""
    if not selected_log:
        return gr.update(value="")
    filepath = _get_user_filepath(request.username, USER_PROMPTS_FILE)
    prompts = _load_json(filepath, default=[])
    match = _find_prompt_entry(prompts, selected_log)
    if not match:
        return gr.update(value="")
    system_prompt = match.get("system_prompt", match.get("prompt", ""))
    return gr.update(value=system_prompt)


def load_system_prompt_confirm(request: gr.Request, selected_log: str):
    """Loads a system prompt after user confirmation and closes the prompt panel."""
    if not selected_log:
        gr.Warning("Please select a system prompt.")
        return gr.update(value=""), gr.update(visible=False)
    filepath = _get_user_filepath(request.username, USER_PROMPTS_FILE)
    prompts = _load_json(filepath, default=[])
    match = _find_prompt_entry(prompts, selected_log)
    if not match:
        gr.Warning("Selected system prompt not found.")
        return gr.update(value=""), gr.update(visible=False)
    system_prompt = match.get("system_prompt", match.get("prompt", ""))
    return gr.update(value=system_prompt), gr.update(visible=False)


def refresh_system_prompts(request: gr.Request):
    filepath = _get_user_filepath(request.username, USER_PROMPTS_FILE)
    prompts = _load_json(filepath, default=[])
    return gr.update(choices=_build_prompt_choices(prompts), value=None)


def validate_interaction(text, llm, top_k, temp, top_p, index_name, task_config, audio_qa_mode=None,
                         text_llm_name=None):
    """Validates playground inputs before sending a query to the LLM."""
    if not llm: raise gr.Error("Please select an LLM.")
    if not task_config: raise gr.Error("Please select a Task Configuration.")
    task_config_dict = json.loads(task_config)
    model_interface = llm_handler.available_llms.get(llm, {}).get("interface")
    if model_interface != "audio":
        if not text.strip():
            raise gr.Error("Query cannot be empty.")
    else:
        if not dynamic_data.get("audio_buffer"):
            raise gr.Error("No audio recorded. Please record audio first.")
        if task_config_dict.get("audio_mode") == "qa" and audio_qa_mode == "whisper_llm":
            if not text_llm_name:
                raise gr.Error("Please select a Text LLM for Whisper + LLM mode.")

    if task_config_dict.get("RAG") and not index_name:
        raise gr.Error("An index must be selected for this RAG task.")

    # Parameter validation
    if not (isinstance(top_k, (int, float)) and 0 <= top_k <= 10): raise gr.Error(
        "K must be a number between 0 and 10.")
    if not (isinstance(temp, (int, float)) and 0 <= temp <= 2): raise gr.Error("Temperature must be between 0 and 2.")
    if not (isinstance(top_p, (int, float)) and 0 <= top_p <= 1): raise gr.Error("Top-p must be between 0 and 1.")


def summarize_conversation(
    history,
    llm_name,
    task_config_str,
    system_prompt,
    temp,
    top_p,
    max_tokens,
    request: gr.Request,
):
    """Generates a summary for the full chat history using the selected LLM."""
    if not llm_name:
        raise gr.Error("Please select an LLM.")
    if not history:
        raise gr.Error("No conversation to summarize.")

    task_config = json.loads(task_config_str) if task_config_str else settings.BASIC_CONFIG
    if task_config.get("interface") != "text":
        raise gr.Error("Summarization is only supported for text tasks.")

    if task_config.get("name") == "DM":
        payload = llm_handler.get_llm_generator(
            llm_name,
            task_name="DM",
        ).end(_dialogue_manager_session_id(request))
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise gr.Error("The dialogue manager returned an empty summary.")
        return summary.strip()

    turns = []
    for user_msg, assistant_msg in history:
        user_text = remove_html_tags(user_msg or "")
        assistant_text = remove_html_tags(assistant_msg or "")
        turns.append(f"User: {user_text}\nAssistant: {assistant_text}")
    conversation_text = "\n\n".join(turns).strip()
    if not conversation_text:
        raise gr.Error("No conversation to summarize.")

    query = (
        "Summarize the following conversation in a concise paragraph:\n\n"
        f"{conversation_text}"
    )

    summary = ""
    stream = _process_llm_request(
        llm_name,
        system_prompt,
        [],
        query,
        0,
        "",
        task_config,
        dynamic_data["retriever_instance"],
        temperature=temp,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    for updated_history, _documents in stream:
        if updated_history:
            summary = updated_history[-1][1]

    return summary


def _collect_llm_response(response):
    if isinstance(response, str):
        return response
    try:
        return "".join([part for part in response])
    except TypeError:
        return str(response)

def _transcribe_audio_for_retrieval(audio, language=None):
    whisper_name = next(
        (
            name
            for name, entry in llm_handler.available_llms.items()
            if entry.get("interactor") == "whisper"
        ),
        None,
    )
    if whisper_name is None:
        raise ValueError(
            "SQA Salamandra retrieval requires a configured Whisper model."
        )

    transcription = _collect_llm_response(
        llm_handler(
            whisper_name,
            "",
            [],
            [],
            audio=audio,
            language=language,
        )
    ).strip()
    if not transcription:
        raise ValueError(
            "Whisper returned an empty transcription for RAG retrieval."
        )
    return transcription

def interact(history, input_text, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config_str, language=None, audio_qa_mode=None, text_llm_name=None, request: gr.Request = None):
    """Handles user interaction in the Gradio chat interface."""
    task_config = json.loads(task_config_str)
    history = history or []

    # Handle audio interface if needed
    audio_in = None
    retrieval_query = None
    model_entry = llm_handler.available_llms.get(llm_name, {})
    model_interface = model_entry.get("interface")
    interactor = model_entry.get("interactor")
    if model_interface == "audio":
        audio_in = bytes(deepcopy(dynamic_data.get("audio_buffer", [])))
        dynamic_data["audio_buffer"] = []
        audio_mode = task_config.get("audio_mode")
        if task_config.get("name") == "RAG" and interactor in {
            "sqa_salamandra_2b",
            "sqa_salamandra_7b",
        }:
            input_text = "Give an answer to the spoken query."
            retrieval_query = _transcribe_audio_for_retrieval(
                audio_in,
                language=language,
            )
        elif audio_mode == "transcription":
            input_text = "Transcribe the audio."
        elif audio_mode == "diarization":
            input_text = "Diarize the audio."
        elif audio_mode == "qa":
            if audio_qa_mode == "whisper_llm":
                transcription = _collect_llm_response(
                    llm_handler(
                        llm_name,
                        "",
                        [],
                        [],
                        audio=audio_in,
                        language=language,
                    )
                )
                input_text = transcription
                task_config = {"interface": "text", "RAG": False, "service": "local",
                               "name": "Audio QA (Whisper + LLM)"}
                llm_name = text_llm_name
                audio_in = None
            else:
                input_text = ""
        else:
            input_text = "Transcribe and respond to the audio."

    session_id = None
    if task_config.get("name") == "DM":
        session_id = _dialogue_manager_session_id(request)

    stream = _process_llm_request(
        llm_name, system_prompt, history, input_text, docs_k, index_name,
        task_config, dynamic_data["retriever_instance"],
        temperature=temp, top_p=top_p, max_tokens=max_tokens, audio=audio_in,
        language=language, retrieval_query=retrieval_query, session_id=session_id,
    )

    for updated_history, documents in stream:
        documents_html = [markdown.markdown(d) for d in documents]
        context_html = context_html_template.render(documents=documents_html)
        yield (
            updated_history,
            context_html,
            gr.update(visible=bool(task_config.get("RAG"))),
            gr.Textbox(value="", interactive=False),
        )


# --- Feedback Tab ---
def _load_feedback_df(force_reload: bool = False) -> pd.DataFrame:
    """Loads the user feedback data into a pandas DataFrame."""
    if not force_reload and dynamic_data.get("feedback_df") is not None:
        return dynamic_data["feedback_df"]

    filepath = os.path.join(settings.USER_WORKSPACES, USER_FEEDBACK_FILE)
    data = _load_json(filepath, default=[])
    df = pd.DataFrame(data)
    dynamic_data["feedback_df"] = df
    return df


def _get_feedback_df():
    """Returns Gradio components for the feedback tab."""
    df = _load_feedback_df(force_reload=True)
    cols = ["None"] + list(df.columns) if not df.empty else ["None"]
    return gr.Dataframe(df, interactive=False), gr.Dropdown(label="Filter Columns", choices=cols, value="None")


def save_feedback(request: gr.Request, binary_feedback: str, chatbot: List, system_prompt: str, rag_html: str,
                  model_name: str, custom_feedback: str):
    """Saves user feedback to the shared feedback file."""
    message = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user": request.username,
        "feedback": binary_feedback,
        "custom_feedback": custom_feedback,
        "model": model_name,
        "system_prompt": system_prompt,
        "retrieved_docs": extract_docs_from_rendered_template(rag_html),
        "generated_response": remove_html_tags(chatbot[-1][1]),
        "full_history": [[remove_html_tags(msg) for msg in turn] for turn in chatbot]
    }

    filepath = os.path.join(settings.USER_WORKSPACES, USER_FEEDBACK_FILE)
    all_feedback = _load_json(filepath)
    all_feedback.append(message)
    _save_json(filepath, all_feedback)

    gr.Info("Feedback saved. Thank you!")
    return gr.update(visible=False)


# --- Dynamic Component Loaders (prefixed with '_get_') ---
def _get_task_configs():
    configs = []
    for fn in os.listdir(settings.TASK_CONFIG_DIR):
        filepath = os.path.join(settings.TASK_CONFIG_DIR, fn)
        content = _load_json(filepath, default={})
        if "name" in content:
            configs.append((content["name"], json.dumps(content)))
    preferred_order = ["Basic LLM", "SDialog", "Summarization", "RAG", "DM", "Audio QA", "Transcription", "Diarization"]
    order_index = {name: idx for idx, name in enumerate(preferred_order)}
    configs.sort(key=lambda item: order_index.get(item[0], len(preferred_order)))
    default_value = None
    for label, value in configs:
        if label == "Basic LLM":
            default_value = value
            break
    if default_value is None and configs:
        default_value = configs[0][1]
    return gr.Radio(label="Task configuration", choices=configs, value=default_value), configs


def _get_retrievers(user: str):
    with open(settings.RETRIEVER_CONFIG_PATH) as f:
        retrievers = json.load(f)
    user_retriever_path = _get_user_filepath(user, USER_RETRIEVERS_FILE)
    user_retrievers = _load_json(user_retriever_path, default={})
    retrievers.update(user_retrievers)
    print("RETRIEBERS", str(retrievers))
    return gr.Radio(label="Vector Store", choices=[(k, v) for k, v in retrievers.items()]), retrievers


def _build_prompt_choices(prompts):
    choices = []
    for idx in range(len(prompts) - 1, -1, -1):
        entry = prompts[idx]
        label = entry.get("name", f"Prompt {idx + 1}")
        choices.append((label, str(idx)))
    return choices


def _get_prompts(user: str):
    filepath = _get_user_filepath(user, USER_PROMPTS_FILE)
    prompts = _load_json(filepath)
    return gr.Radio(label="Saved System Prompts", choices=_build_prompt_choices(prompts))


def _get_historical_prompts(user: str):
    filepath = _get_user_filepath(user, USER_HISTORY_FILE)
    logs = _load_json(filepath)
    return gr.Radio(label="Saved Histories", choices=_build_history_choices(logs))


def _get_online_models(available_llms):
    probe_models = str(os.environ.get("ELOQ_PROBE_MODELS_ON_LOAD", "true")).strip().lower() in {"1", "true", "yes",
                                                                                                "on"}

    def _interactor_name(model_name: str) -> str:
        entry = available_llms.get(model_name, {})
        return str(entry.get("interactor", "")).strip().lower()

    def _is_sdialog_model(model_name: str) -> bool:
        return _interactor_name(model_name) == "sdialog"

    def _is_dialogue_manager_model(model_name: str) -> bool:
        return _interactor_name(model_name) == "dialogue_manager"

    def _is_model_online(model_name):
        import requests

        logger.info("Checking availability of %s", model_name)
        entry = available_llms[model_name]
        api_endpoint = entry["api_endpoint"].rstrip("/")
        base_endpoint = (
            api_endpoint[:-3]
            if api_endpoint.endswith("/v1")
            else api_endpoint
        )
        headers = {}
        if entry.get("api_key"):
            headers["Authorization"] = f"Bearer {entry['api_key']}"

        if _is_dialogue_manager_model(model_name):
            check_urls = [f"{base_endpoint}/docs"]
        else:
            check_urls = [
                f"{base_endpoint}/health",
                f"{api_endpoint}/models",
            ]
        for check_url in check_urls:
            try:
                response = requests.get(
                    check_url,
                    headers=headers,
                    timeout=5,
                )
                if response.ok:
                    logger.info(
                        "Model %s is available via %s",
                        model_name,
                        check_url,
                    )
                    return True
            except requests.RequestException as exc:
                logger.debug(
                    "Availability check failed for %s at %s: %s",
                    model_name,
                    check_url,
                    exc,
                )

        logger.warning("Model %s did not pass an availability check", model_name)
        return False

    choices = [(llm, llm) for llm in available_llms.keys()]
    if probe_models:
        max_workers = min(8, max(1, len(choices)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            probe_results = executor.map(
                _is_model_online,
                [choice[1] for choice in choices],
            )
            probed_online_choices = [
                choice
                for choice, is_online in zip(choices, probe_results)
                if is_online
            ]
    else:
        # Avoid cross-endpoint health probes on page load unless explicitly enabled.
        probed_online_choices = choices

    # Preserve all online models (including task-specific ones) for task-level filtering.
    dynamic_data["online_llms"] = [c[1] for c in probed_online_choices]

    # The initial task is Basic LLM, so initialize its selector with text models.
    online_choices = [
        choice
        for choice in probed_online_choices
        if (
            not _is_sdialog_model(choice[1])
            and available_llms[choice[1]].get("interface") == "text"
        )
    ]
    default_model = online_choices[0][1] if online_choices else None
    return gr.Radio(label="Available LLMs", choices=online_choices, value=default_model), dynamic_data["online_llms"]


def update_llm_choices(task_config_str: str, audio_qa_mode: str | None = None) -> gr.update:
    """Update LLM choices based on task interface."""
    task_config = json.loads(task_config_str) if task_config_str else {}
    task_name = task_config.get("name")
    if task_name == "RAG":
        return gr.update(
            label="Available LLMs",
            choices=[],
            value=None,
            visible=False,
        )

    interface = task_config.get("interface", "text")
    interfaces = interface if isinstance(interface, list) else [interface]
    audio_mode = task_config.get("audio_mode")

    def _interactor_name(model_name: str) -> str:
        entry = llm_handler.available_llms.get(model_name, {})
        return str(entry.get("interactor", "")).strip().lower()

    def _is_whisper_model(model_name: str) -> bool:
        return _interactor_name(model_name) == "whisper"

    def _is_meusli_model(model_name: str) -> bool:
        return _interactor_name(model_name) == "meusli"

    def _is_whisperx_model(model_name: str) -> bool:
        return _interactor_name(model_name) == "whisperx"

    def _is_sdialog_model(model_name: str) -> bool:
        return _interactor_name(model_name) == "sdialog"

    def _is_dialogue_manager_model(model_name: str) -> bool:
        return _interactor_name(model_name) == "dialogue_manager"

    def _is_eligible(model_name):
        if not check_llm_interface(
            model_name,
            interface,
            available_llms=llm_handler.available_llms,
        ):
            return False

        if task_name == "DM":
            return _is_dialogue_manager_model(model_name)
        if _is_dialogue_manager_model(model_name):
            return False

        if task_name == "SDialog":
            if not _is_sdialog_model(model_name):
                return False
        elif _is_sdialog_model(model_name):
            return False

        if interfaces != ["audio"] or not audio_mode:
            return True

        if audio_mode == "transcription":
            return (
                _is_whisper_model(model_name)
                or _is_meusli_model(model_name)
            ) and not _is_whisperx_model(model_name)
        if audio_mode == "diarization":
            return _is_whisperx_model(model_name)
        if audio_mode == "qa":
            if audio_qa_mode == "whisper_llm":
                return _is_whisper_model(model_name) and not _is_whisperx_model(model_name)
            return not _is_whisper_model(model_name) and not _is_whisperx_model(model_name)
        return True

    choices = [
        (name, name)
        for name in dynamic_data.get("online_llms", [])
        if _is_eligible(name)
    ]

    default_value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=default_value, visible=True)

def update_rag_llm_choices(task_config_str: str):
    task_config = json.loads(task_config_str) if task_config_str else {}
    if task_config.get("name") != "RAG":
        hidden = gr.update(choices=[], value=None, visible=False)
        return hidden, hidden

    text_choices = []
    speech_choices = []
    for model_name in dynamic_data.get("online_llms", []):
        entry = llm_handler.available_llms.get(model_name, {})
        interactor = str(entry.get("interactor", "")).strip().lower()

        if (
            entry.get("interface") == "text"
            and interactor not in {"sdialog", "dialogue_manager"}
        ):
            text_choices.append((model_name, model_name))
        elif interactor in {
            "sqa_salamandra_2b",
            "sqa_salamandra_7b",
        }:
            speech_choices.append((model_name, model_name))

    return (
        gr.update(
            choices=text_choices,
            value=None,
            visible=True,
        ),
        gr.update(
            choices=speech_choices,
            value=None,
            visible=True,
        ),
    )

def select_rag_text_model(model_name, task_config_str):
    task_config = json.loads(task_config_str) if task_config_str else {}
    if task_config.get("name") != "RAG" or not model_name:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    dynamic_data["audio_buffer"] = []
    return (
        gr.update(choices=[model_name], value=model_name),
        gr.update(value=None),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(
            choices=LANGUAGE_CHOICES,
            value="en",
            visible=False,
        ),
    )

def select_rag_speech_model(
    model_name,
    task_config_str,
    audio_qa_mode: str | None = None,
):
    task_config = json.loads(task_config_str) if task_config_str else {}
    task_name = task_config.get("name")
    if not model_name:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    return (
        gr.update(choices=[model_name], value=model_name),
        gr.update(value=None),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(
            choices=[("Spanish", "es")],
            value="es",
            visible=True,
        ),
    )

def update_text_llm_choices(task_config_str: str, audio_qa_mode: str | None = None) -> gr.update:
    task_config = json.loads(task_config_str) if task_config_str else {}
    interface = "audio" if task_config.get("interface") == "audio" else "text"
    audio_mode = task_config.get("audio_mode")

    def _interactor_name(model_name: str) -> str:
        entry = llm_handler.available_llms.get(model_name, {})
        return str(entry.get("interactor", "")).strip().lower()

    if not (interface == "audio" and audio_mode == "qa" and audio_qa_mode == "whisper_llm"):
        return gr.update(visible=False, value=None)

    filtered_choices = [
        (name, name)
        for name in dynamic_data.get("online_llms", [])
        if (
            llm_handler.available_llms.get(name, {}).get("interface") == "text"
            and _interactor_name(name) not in {"sdialog", "dialogue_manager"}
        )
    ]

    default_value = filtered_choices[0][1] if filtered_choices else None
    return gr.update(choices=filtered_choices, value=default_value, visible=True)


def update_llm_params_visibility(task_config_str: str, audio_qa_mode: str | None = None) -> gr.update:
    task_config = json.loads(task_config_str) if task_config_str else {}
    if task_config.get("name") == "DM":
        return gr.update(visible=False)

    interface = "audio" if task_config.get("interface") == "audio" else "text"
    audio_mode = task_config.get("audio_mode")

    if interface == "text":
        return gr.update(visible=True)
    if interface == "audio" and audio_mode == "qa" and audio_qa_mode == "whisper_llm":
        return gr.update(visible=True)
    return gr.update(visible=False)


def update_rag_params_visibility(task_config_str: str) -> gr.update:
    task_config = json.loads(task_config_str) if task_config_str else {}
    rag_enabled = task_config.get("name") == "RAG"
    return gr.update(visible=rag_enabled)
