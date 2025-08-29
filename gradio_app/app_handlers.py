import shutil
import datetime
import os
import json
from typing import List
import logging
from copy import deepcopy


import markdown
import gradio as gr
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.task_handlers import get_task_handler
from gradio_app.backend.query_llm import LLMHandler
from gradio_app.helpers import replace_doc_links, _load_json, _save_json, remove_html_tags, extract_docs_from_rendered_template, _get_user_filepath
from retrievers.client import RetrieverClient
from settings import settings, USER_FEEDBACK_FILE, USER_HISTORY_FILE, USER_PROMPTS_FILE, USER_RETRIEVERS_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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

def perform_ingest(index_name: str, chunk_size: int, percentile: int, embed_name: str, file_paths: List[str], splitting_strategy: str, retriever_address: str):
    """Handles the document ingestion process."""
    if not file_paths:
        raise gr.Error("You must upload at least one file.")
    if isinstance(file_paths, str):
        file_paths = file_paths.split(",")
    
    gr.Info("Ingesting documents...")
    retriever = RetrieverClient(endpoint=retriever_address)
    uploaded_files = [os.path.join(settings.GENERIC_UPLOAD, os.path.basename(fp)) for fp in file_paths]
    
    retriever.create_vs(
        uploaded_files,
        chunk_size,
        percentile,
        embed_name,
        index_name,
        splitting_strategy
    )
    
    shutil.rmtree(settings.GENERIC_UPLOAD, ignore_errors=True)
    gr.Info("Ingestion successful!")

def _process_llm_request(llm_name, system_prompt, history, query, docs_k, index_name, task_config, retriever_instance, **kwargs):
    """
    Core generator function to handle LLM requests. Yields response parts and documents.
    This is the central logic used by both Gradio and FastAPI.
    """
    task_handler = get_task_handler(task_config, llm_handler, retriever_instance)
    
    if not query:
        raise ValueError("Empty query submitted.")

    history = history + [[query, ""]]
    
    # Extract audio data if present in kwargs
    audio_data = kwargs.get("audio")

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
        audio=audio_data
    )

    for part, documents in stream:
        history[-1][1] += part
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

def validate_ingestion_inputs(index_name, embedder, files, chunk_length, percentile, retriever_addr):
    """Validates all inputs on the Ingestion tab before starting the process."""
    if not all([index_name, embedder, files, retriever_addr]):
        raise gr.Error("Please fill all required fields: Index Name, Embedder, Vector Store, and upload at least one file.")
    
    for f in files:
        if not any(f.name.endswith(suff) for suff in settings.SUPPORTED_FILE_TYPES):
            raise gr.Error(f"File '{os.path.basename(f.name)}' has an unsupported file type.")

    try:
        if not 0 < int(chunk_length) <= 2000: raise ValueError()
    except ValueError:
        raise gr.Error("'Chunk Length' must be an integer between 1 and 2000.")
    
    try:
        if not 0 < int(percentile) <= 100: raise ValueError()
    except ValueError:
        raise gr.Error("'Percentile' must be an integer between 1 and 100.")

# --- Playground Tab ---
def get_dynamic_components(request: gr.Request) -> tuple:
    """Loads all dynamic data (prompts, history, etc.) when the UI loads."""
    user = request.username
    feedback_df, avail_cols = _get_feedback_df()
    task_configs_radio, _ = _get_task_configs()
    retrievers_radio, _ = _get_retrievers(user)
    online_choices, _ = _get_online_models(llm_handler.available_llms)
    
    return (
        task_configs_radio,
        _get_prompts(user),
        _get_historical_prompts(user),
        online_choices,
        retrievers_radio,
        retrievers_radio, # For both Playground and Ingestion tabs
        feedback_df,
        avail_cols
    )

def change_retriever(selected_retr_endpoint: str) -> gr.Radio:
    """Updates the available indexes when a different Vector Store is selected."""
    dynamic_data["retriever_instance"] = RetrieverClient(endpoint=selected_retr_endpoint)
    choices = dynamic_data["retriever_instance"].list_vs()
    return gr.Radio(label="Index name", choices=choices)

def save_prompt(request: gr.Request, prompt: str):
    """Saves a new system prompt for the current user."""
    if not prompt:
        gr.Warning("Cannot save an empty prompt.")
        return
    filepath = _get_user_filepath(request.username, USER_PROMPTS_FILE)
    prompts = _load_json(filepath)
    prompts.append({"name": f"{prompt[:20]}...", "prompt": prompt})
    _save_json(filepath, prompts)
    gr.Info("Prompt saved successfully!")
    # To refresh the dropdown, we would need to return a new gr.Dropdown object
    # For simplicity, user needs to reload to see the new prompt.

def store_history(request: gr.Request, history: List[List[str]], prompt: str):
    """Saves the current conversation history for the user."""
    filepath = _get_user_filepath(request.username, USER_HISTORY_FILE)
    logs = _load_json(filepath)
    log_name = f"{datetime.date.today()} - {prompt[:15] if prompt else history[0][0][:15]}..."
    logs.append({"history": history, "prompt": prompt, "name": log_name})
    _save_json(filepath, logs)
    gr.Info(f"History saved for user '{request.username}'.")

def validate_interaction(text, llm, top_k, temp, top_p, index_name, task_config):
    """Validates playground inputs before sending a query to the LLM."""
    if not text.strip(): raise gr.Error("Query cannot be empty.")
    if not llm: raise gr.Error("Please select an LLM.")
    if not task_config: raise gr.Error("Please select a Task Configuration.")
    
    task_config_dict = json.loads(task_config)
    if task_config_dict.get("RAG") and not index_name:
        raise gr.Error("An index must be selected for this RAG task.")
    
    # Parameter validation
    if not (isinstance(top_k, (int, float)) and 0 <= top_k <= 10): raise gr.Error("K must be a number between 0 and 10.")
    if not (isinstance(temp, (int, float)) and 0 <= temp <= 2): raise gr.Error("Temperature must be between 0 and 2.")
    if not (isinstance(top_p, (int, float)) and 0 <= top_p <= 1): raise gr.Error("Top-p must be between 0 and 1.")

def interact(history, input_text, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config_str):
    """Handles user interaction in the Gradio chat interface."""
    task_config = json.loads(task_config_str)
    history = history or []
    
    # Handle audio interface if needed
    audio_in = None
    if task_config.get("interface") == "audio":
        audio_in = deepcopy(dynamic_data.get("audio_buffer", []))
        dynamic_data["audio_buffer"] = []
        input_text = "Transcribe and respond to the audio."

    stream = _process_llm_request(
        llm_name, system_prompt, history, input_text, docs_k, index_name,
        task_config, dynamic_data["retriever_instance"],
        temperature=temp, top_p=top_p, max_tokens=max_tokens, audio=audio_in
    )
    
    for updated_history, documents in stream:
        documents_html = [markdown.markdown(d) for d in documents]
        context_html = context_html_template.render(documents=documents_html)
        yield (
            updated_history,
            context_html,
            gr.update(visible=len(documents) > 0),
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

def save_feedback(request: gr.Request, binary_feedback: str, chatbot: List, system_prompt: str, rag_html: str, model_name: str, custom_feedback: str):
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
    return gr.Radio(label="Task configuration", choices=configs), configs

def _get_retrievers(user: str):
    with open(settings.RETRIEVER_CONFIG_PATH) as f:
        retrievers = json.load(f)
    user_retriever_path = _get_user_filepath(user, USER_RETRIEVERS_FILE)
    user_retrievers = _load_json(user_retriever_path, default={})
    retrievers.update(user_retrievers)
    return gr.Radio(label="Vector Store", choices=[(k, v) for k, v in retrievers.items()]), retrievers

def _get_prompts(user: str):
    filepath = _get_user_filepath(user, USER_PROMPTS_FILE)
    prompts = _load_json(filepath)
    return gr.Dropdown(label="Prompt", choices=[(p["name"], p["prompt"]) for p in prompts])

def _get_historical_prompts(user: str):
    filepath = _get_user_filepath(user, USER_HISTORY_FILE)
    logs = _load_json(filepath)
    return gr.Dropdown(label="History", choices=[p["name"] for p in logs])

def _get_online_models(available_llms):
    def _is_model_online(model_name):
        return True
        gr.Info(f"Checking availability of {model_name}")
        task_handler = get_task_handler(settings.BASIC_CONFIG, llm_handler, dynamic_data["retriever_instance"])
        query = history_user_entry = "hello, say one random words"
        history = [[history_user_entry, ""]]
        try: 
            for part, documents in task_handler(model_name,
                                                "",
                                                history,
                                                query,
                                                0,
                                                "index_name",
                                                max_tokens=2):
                return True
        except Exception as e:
            return False

    choices = [(llm, llm) for llm in available_llms.keys()]
    online_choices = [choice for choice in choices if _is_model_online(choice[1])]
    return gr.Radio(label="Available LLMs", choices=online_choices), [c[1] for c in online_choices]
