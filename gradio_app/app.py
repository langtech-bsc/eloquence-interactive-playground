import logging
import os
import json
from copy import deepcopy
import shutil
import tempfile
import sqlite3
from typing import *

import gradio as gr
import markdown

from fastapi import FastAPI, UploadFile, Form, File
from pydantic import TypeAdapter
from fastapi.middleware.cors import CORSMiddleware

from gradio_app.helpers import  extract_docs_from_rendered_template
from gradio_app.messages import *
from retrievers.client import RetrieverClient
from settings import settings, normalize_path_prefix
from gradio_app.app_handlers import (
    _get_task_configs,
    _process_llm_request,
    validate_interaction,
    interact,
    change_retriever,
    save_system_prompt,
    load_system_prompt_preview,
    load_system_prompt_confirm,
    refresh_system_prompts,
    store_history,
    load_history,
    load_history_preview,
    load_history_confirm,
    summarize_conversation,
    save_feedback,
    upload_file_for_ingest,
    validate_ingestion_inputs,
    perform_ingest,
    get_dynamic_components,
    _get_online_models,
    _load_feedback_df,
    dynamic_data,
    llm_handler,
    load_task,
    update_llm_choices,
    update_text_llm_choices,
    update_llm_params_visibility,
    update_rag_params_visibility,
)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubpathMiddleware:
    def __init__(self, app, *, settings):
        self.app = app
        self.settings = settings

    def _decode_headers(self, scope):
        headers = {}
        for name, value in scope.get("headers", []):
            try:
                header_name = name.decode("latin-1").lower()
                header_value = value.decode("latin-1")
            except UnicodeDecodeError:
                continue
            headers[header_name] = header_value
        return headers

    def _header_prefix(self, headers):
        for header_name in self.settings.PATH_PREFIX_HEADERS:
            header_value = headers.get(header_name)
            if header_value:
                normalized = normalize_path_prefix(header_value)
                if normalized:
                    return normalized
        return ""

    def _match_path_prefix(self, path: str) -> str:
        for prefix in self.settings.PATH_PREFIXES:
            if prefix and path.startswith(prefix):
                return prefix
        return ""

    def _strip_prefix(self, scope, prefix: str):
        path = scope.get("path", "")
        stripped = path[len(prefix):]
        if not stripped:
            stripped = "/"
        elif not stripped.startswith("/"):
            stripped = f"/{stripped}"
        scope["path"] = stripped

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = self._decode_headers(scope)
        prefix = self._header_prefix(headers)
        if not prefix:
            prefix = self._match_path_prefix(scope.get("path", ""))
        if not prefix:
            prefix = normalize_path_prefix(self.settings.ROOT_PATH)

        if prefix and scope.get("path", "").startswith(prefix):
            self._strip_prefix(scope, prefix)

        scope["root_path"] = prefix or ""
        await self.app(scope, receive, send)

def show_feedback(request: gr.Request, x: gr.LikeData):
    # This is currently not used:  x.index[0], x.index[1]
    return gr.update(visible=True), gr.update(value=str(x.liked))

def process_filter_value_change(selected_col: str, selected_val: str):
    feedback_df = _load_feedback_df()
    if selected_col == "None":
        val_dropdown = gr.Dropdown(interactive=False)
    else:
        avail_choices = ["all"] + list(feedback_df[selected_col].unique())
        selected_val = selected_val if (selected_val is not None and selected_val != "all" and selected_val in avail_choices) else "all"
        val_dropdown = gr.Dropdown(choices=avail_choices, value=selected_val, interactive=True)
    if selected_col == "None" or selected_val == "all":
        filtered_df = feedback_df
    else:
        filtered_df = feedback_df[feedback_df[selected_col]==selected_val]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    filtered_df.to_json(temp_file.name, orient="records", indent=2)
    temp_file.close()
 
    return gr.DataFrame(filtered_df, interactive=False), gr.update(value=temp_file.name), val_dropdown

def authenticate(user: str, password: str) -> bool:
    """Authenticates a user against the SQLite database."""
    logging.info(f"Attempting to authenticate user: {user}")
    logging.info(f"Using database: {settings.SQL_DB}")
    with sqlite3.connect(settings.SQL_DB) as conn:
        cursor = conn.cursor()
        result = cursor.execute(
            "SELECT username FROM users WHERE username=? AND password=?", (user, password)
        ).fetchone()
    success = result and result[0] == user
    logger.info(f"Login attempt for user '{user}': {'Success' if success else 'Failed'}")
    return success


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SubpathMiddleware, settings=settings)

async def query_llm_general(available_llms, audio_file=None, **kwargs):
    """General purpose LLM query handler for the API."""
    if kwargs["llm_name"] not in available_llms.keys():
        return "", []  # Return empty text and documents for unknown LLM
    
    _, task_configs_list = _get_task_configs()
    task_configs = {k: json.loads(v) for k, v in task_configs_list}
    task_config = task_configs.get(kwargs.get("task_config"), settings.BASIC_CONFIG)
    
    retriever_address = kwargs.get("retriever_address", settings.RETRIEVER_ENDPOINT)
    retriever_instance = RetrieverClient(endpoint=retriever_address)
    
    history = kwargs.get("history") or []
    query = kwargs.get("input_text", "")
    audio_data = await audio_file.read() if audio_file else None

    if task_config.get("interface") == "audio" and audio_data:
        query = "Describe the audio."
    
    stream = _process_llm_request(
        kwargs["llm_name"], kwargs.get("system_prompt"), history, query,
        kwargs.get("docs_k"), kwargs.get("index_name"), task_config, retriever_instance,
        temperature=kwargs.get("temp"), top_p=kwargs.get("top_p"),
        max_tokens=kwargs.get("max_tokens"), audio=audio_data, language=kwargs.get("language")
    )
    
    final_text = ""
    final_docs = []
    for updated_history, docs in stream:
        final_text = updated_history[-1][1]
        if docs:
            final_docs = [d if isinstance(d, str) else str(d) for d in docs]
        else:
            final_docs = []
    
    return final_text, final_docs


@app.post("/stream")
async def upload_audio_chunk(audio_chunk: UploadFile):
    """Accepts a streaming audio chunk and adds it to the buffer."""
    data = await audio_chunk.read()
    logger.info("Audio chunk received: %d bytes", len(data))
    dynamic_data["audio_buffer"].extend(data)
    return {"status": "ok", "size_received": len(data)}


@app.post("/stream_finalize")
async def upload_audio_final(audio_file: UploadFile):
    """Accepts the final recording blob and replaces the buffer."""
    data = await audio_file.read()
    logger.info("Final audio received: %d bytes, content_type=%s", len(data), audio_file.content_type)
    if not data:
        return {"status": "error", "size_received": 0}
    dynamic_data["audio_buffer"] = bytearray(data)
    return {"status": "ok", "size_received": len(data)}


@app.post("/query", response_model=ResponseQueryLLM)
async def query_llm_endpoint(body: str = Form(...), audio_file: Optional[UploadFile] = File(None)):
    """Primary endpoint for submitting a single query to the LLM."""
    request_data = TypeAdapter(RequestQueryLLM).validate_json(body)
    try:
        response_text, documents = await query_llm_general(
            available_llms=llm_handler.available_llms,
            audio_file=audio_file,
            **request_data.model_dump()
        )
        if len(documents) < 2:
            documents = []
        return ResponseQueryLLM(text=response_text, documents=documents)
    except Exception as exc:
        logger.exception("Query failed")
        return ResponseQueryLLM(text="", documents=[], error=str(exc))


@app.post("/batch_query", response_model=ResponseBatchQuery)
async def batch_query_endpoint(body: str = Form(...), data_file: UploadFile = File(...)):
    """Processes a batch of conversations from an uploaded JSON file."""
    request_data = TypeAdapter(RequestBatchQuery).validate_json(body)
    batch_data = json.loads(await data_file.read())
    processed_data = {}

    for conv_id, turns in batch_data.items():
        history = []
        for turn in turns:
            response_text, _ = await query_llm_general(available_llms=llm_handler.available_llms, input_text=turn, history=deepcopy(history), **request_data.model_dump())
            history.append([turn, response_text])
        processed_data[conv_id] = history
    
    return ResponseBatchQuery(processed=processed_data)


@app.post("/ingest", response_model=ResponseIngest)
async def ingest_endpoint(content_file: UploadFile, body: str = Form(...)):
    """Handles document ingestion into a specified vector store."""
    request = TypeAdapter(RequestIngest).validate_json(body)
    try:
        # Use a temporary file to safely handle the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=content_file.filename) as tmp:
            shutil.copyfileobj(content_file.file, tmp.file)
            temp_file_path = tmp.name
        # Call the same ingestion logic used by the UI
        perform_ingest(
            index_name=request.index_name,
            chunk_size=request.chunk_size,
            percentile=request.percentile,
            embed_name=request.embed_name,
            file_paths=[temp_file_path], # Pass base name
            splitting_strategy=request.splitting_strategy,
            retriever_address=request.retriever_address
        )
        try:
            os.unlink(temp_file_path)
        except FileNotFoundError:
            pass
        return ResponseIngest(status="success", msg="Document ingested successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return ResponseIngest(status="error", msg=str(e))


@app.get("/list_vs", response_model=ResponseList)
async def list_vector_stores(retriever_address: str = "public"):
    """Lists available Vector Stores (indexes)."""
    if retriever_address == "public":
        retriever_address = settings.RETRIEVER_ENDPOINT
    retriever = RetrieverClient(endpoint=retriever_address)
    stores = retriever.list_vs()
    return ResponseList(available=stores)


@app.get("/list_llms", response_model=ResponseList)
def list_llms():
    """Lists available and online LLM models."""
    _, online_models = _get_online_models(llm_handler.available_llms)
    return ResponseList(available=online_models)


@app.get("/list_embedders", response_model=ResponseList)
def list_embedders():
    """Lists available embedding models."""
    return ResponseList(available=list(settings.EMBEDDING_SIZES.keys()))


@app.get("/retrieval", response_model=ResponseList)
def direct_retrieval(query: str, index_name: str, retriever_address: str = "public", top_k: int = 5):
    """Performs a direct document search in the vector store."""
    if retriever_address == "public":
        retriever_address = settings.RETRIEVER_ENDPOINT
    retriever = RetrieverClient(endpoint=retriever_address)
    docs = retriever.search(index_name=index_name, query=query, top_k=top_k)
    return ResponseList(available=docs)


@app.get("/feedback", response_model=ResponseFeedback)
def get_feedback(filter_column: Optional[str] = None, filter_value: Optional[str] = None):
    """Downloads collected user feedback, with optional filtering."""
    feedback_df = _load_feedback_df(force_reload=True)
    filt_query = ""
    if filter_column and filter_value and filter_column in feedback_df.columns:
        filt_query = f"{filter_column} == '{filter_value}'"
        feedback_df = feedback_df.query(filt_query)

    records = feedback_df.to_dict(orient="records")
    return ResponseFeedback(filter=filt_query, feedback=records)


examples = [
    "What is the goal of Eloquence?",
    "Which countries are participating?",
    "What is Omilia's task in this project?",
    "What computational resources are available?",
]

with gr.Blocks(theme=gr.themes.Monochrome(), css=settings.CSS, js=settings.JS_CODE) as demo:
    # --- PLAYGROUND TAB ---
    with gr.Tab("Playground"):
        with gr.Row():
            # Main chat column
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    avatar_images=('assets/user.jpeg', 'assets/eloq.png'),
                    bubble_full_width=True,
                    height=500,
                    label="EloquenceBot",
                    sanitize_html=False,
                    show_copy_button=False
                )
                
                # Feedback UI
                with gr.Row(visible=False) as additional_feedback:
                    user_binary_feedback = gr.Dropdown(label="Feedback", choices=["False", "True"])
                    user_additional_feedback = gr.Textbox(label="Additional Feedback", lines=2)
                    user_additional_feedback_submit = gr.Button("Submit Feedback")
                
                # Input UI
                with gr.Row(elem_id="input_controls_row"):
                    with gr.Column(visible=True) as text_column:
                        input_textbox = gr.Textbox(
                            scale=3,
                            show_label=False,
                            container=False,
                        )
                    with gr.Column(visible=False) as audio_column:
                        hidden_submit_btn = gr.Button(visible=False, elem_id="trigger_audio_submit")
                        html = """
                        <div class="audio-controls">
                            <div class="audio-record-buttons">
                                <button class="lg secondary  svelte-cmf5ev" onclick="startStreaming()">Start Recording</button>
                                <button class="lg secondary  svelte-cmf5ev" onclick="stopStreaming()">Stop Recording</button>
                            </div>
                            <div class="audio-playback">
                                <audio id="recorded_audio" controls></audio>
                            </div>
                            <div id="recordstatus" class="audio-status"></div>
                        </div>
                        """

                        audio_object = gr.HTML(html)
                        language_dropdown = gr.Dropdown(
                            label="Language",
                            choices=[
                                ("English", "en"),
                                ("Spanish", "es"),
                                ("Italian", "it"),
                                ("Greek", "el"),
                                ("Serbian", "sr"),
                            ],
                            value="en",
                            visible=False
                        )
                    with gr.Column():
                        with gr.Row(elem_id="submit_clear_row"):
                            submit_btn = gr.Button("Submit", elem_id="submit_btn")
                            clear_btn = gr.Button("Clear")
                        summarize_btn = gr.Button("Summarize conversation", visible=False)

                gr.Examples(examples, input_textbox)

            # RAG and settings column
            with gr.Column(scale=1):
                with gr.Accordion("Settings & Configuration", open=False):
                    task_config = gr.Radio(label="Task configuration", elem_id="task_config")
                    audio_qa_mode = gr.Radio(
                        label="Audio QA Mode",
                        choices=[("Whisper + LLM", "whisper_llm"), ("Speech LLM", "speech_llm")],
                        visible=False,
                        elem_id="audio_qa_mode",
                    )
                    llm_name = gr.Radio(label="Available LLMs", visible=False, elem_id="llm_name")
                    text_llm_name = gr.Radio(label="Text LLM", visible=False, elem_id="text_llm_name")
                    retrievers_radio = gr.Radio(label="Vector Store", visible=False)
                    index_name = gr.Radio(label="Index name", visible=False)
                    system_prompt = gr.Textbox(value="", label="System Prompt", lines=4)
                    
                    with gr.Row():
                        load_prompt_btn = gr.Button("Load System Prompt")
                        save_prompt_btn = gr.Button("Save System Prompt")

                    with gr.Row():
                        load_history_btn = gr.Button("Load History")
                        save_btn = gr.Button("Save History")
                
                with gr.Accordion("LLM Parameters", open=False, visible=False) as llm_params_accordion:
                    temp = gr.Slider(0, 2, value=1.0, step=0.1, label="Temperature")
                    top_p = gr.Slider(0, 1, value=0.95, step=0.05, label="Top P")
                    max_tokens = gr.Slider(100, 4000, value=512, step=64, label="Max tokens")
                
                with gr.Accordion("RAG Parameters", open=False, visible=False) as rag_params_accordion:
                    docs_k = gr.Slider(0, 10, value=5, step=1, label="Top K documents")

                summary_box = gr.Textbox(value="", label="Summary", lines=4, interactive=False, visible=False)

                with gr.Column(visible=False, elem_id="prompt_panel") as prompt_panel:
                    gr.Markdown("### Select A System Prompt")
                    prompt_radio = gr.Radio(label="Saved System Prompts", interactive=True, elem_id="prompt_radio")
                    prompt_preview = gr.Textbox(label="Preview", lines=6, interactive=False)
                    with gr.Row():
                        confirm_prompt_btn = gr.Button("Confirm")
                        close_prompt_btn = gr.Button("Close")

                with gr.Column(visible=False, elem_id="history_panel") as history_panel:
                    gr.Markdown("### Select A History")
                    history_radio = gr.Radio(label="Saved Histories", interactive=True, elem_id="history_radio")
                    history_preview = gr.Textbox(label="Preview", lines=8, interactive=False)
                    with gr.Row():
                        confirm_history_btn = gr.Button("Confirm")
                        close_history_btn = gr.Button("Close")
                
                # RAG context display
                with gr.Column(visible=False) as rag_column:
                    gr.Markdown("### Retrieved Context")
                    context_html = gr.HTML()

    # --- INGESTION TAB ---
    with gr.Tab("Ingestion"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Create a New Vector Store Index")
                retrievers_radio_ing = gr.Radio(label="Target Vector Store")
                index_name_ing = gr.Textbox(label="New Index Name")
                embed_name = gr.Radio(choices=list(settings.EMBEDDING_SIZES.keys()), label="Embedder")
                upload_btt = gr.UploadButton("Select Files...", file_count="multiple")
                uploaded_doc_ing = gr.Textbox(label="Selected File(s)", interactive=False)

            with gr.Column():
                gr.Markdown("### Splitting Strategy")
                splitting_strategy = gr.Radio(
                    label="Strategy",
                    choices=[("By-Length (recursive)", "recursive"), ("Semantic", "semantic"), ("By-Length (simple)", "simple")],
                    value="recursive"
                )
                chunk_length = gr.Number(label="Chunk Length (for by-length)", value=500)
                percentile = gr.Number(label="Percentile Threshold (for semantic)", value=95)
                run_ingestion = gr.Button("Run Ingestion")
                ingestion_status = gr.Textbox(label="Ingestion Status", interactive=False, visible=False, elem_id="ingestion_status")


    # --- FEEDBACK TAB ---
    with gr.Tab("Feedback") as feedback_tab:
        with gr.Row():
            filter_column = gr.Dropdown(label="Filter Column")
            filter_value = gr.Dropdown(label="Filter Value", interactive=False)
            download_feedback = gr.DownloadButton(
                        label="Download feedback",
                        value=os.path.join(settings.USER_WORKSPACES, "user_feedback.json"),
                        scale=1,
                        interactive=True,
                    )
        with gr.Row():
            feedback_df = gr.Dataframe()

    demo.load(
        get_dynamic_components,
        [],
        [task_config, prompt_radio, history_radio, llm_name, retrievers_radio, retrievers_radio_ing, feedback_df, filter_column]
    )

    # --- Playground Events ---
    submit_btn.click(
        validate_interaction,
        [input_textbox, llm_name, docs_k, temp, top_p, index_name, task_config, audio_qa_mode, text_llm_name],
        None
    ).success(
        interact,
        [chatbot, input_textbox, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config, language_dropdown, audio_qa_mode, text_llm_name],
        [chatbot, context_html, rag_column, input_textbox]
    ).then(
        lambda: gr.update(interactive=True), None, [input_textbox]
    )

    hidden_submit_btn.click(
        validate_interaction,
        [input_textbox, llm_name, docs_k, temp, top_p, index_name, task_config, audio_qa_mode, text_llm_name],
        None
    ).success(
        interact,
        [chatbot, input_textbox, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config, language_dropdown, audio_qa_mode, text_llm_name],
        [chatbot, context_html, rag_column, input_textbox]
    ).then(
        lambda: gr.update(interactive=True), None, [input_textbox]
    )
    
    clear_btn.click(lambda: ([], "", gr.update(value=None), "", gr.update(visible=False)), [], [chatbot, system_prompt, prompt_radio, summary_box, rag_column])
    summarize_btn.click(
        summarize_conversation,
        [chatbot, llm_name, task_config, system_prompt, temp, top_p, max_tokens],
        [summary_box],
    )
    retrievers_radio.change(change_retriever, [retrievers_radio], [index_name])
    task_config.change(
        load_task,
        [task_config],
        [
            text_column,
            audio_column,
            submit_btn,
            language_dropdown,
            llm_params_accordion,
            audio_qa_mode,
            retrievers_radio,
            index_name,
            rag_column,
            summarize_btn,
            summary_box,
            text_llm_name,
            rag_params_accordion,
        ]
    ).then(
        update_llm_choices,
        [task_config, audio_qa_mode],
        [llm_name],
    ).then(
        update_text_llm_choices,
        [task_config, audio_qa_mode],
        [text_llm_name],
    ).then(
        update_llm_params_visibility,
        [task_config, audio_qa_mode],
        [llm_params_accordion],
    ).then(
        update_rag_params_visibility,
        [task_config],
        [rag_params_accordion],
    )
    audio_qa_mode.change(
        update_llm_choices,
        [task_config, audio_qa_mode],
        [llm_name],
    ).then(
        update_text_llm_choices,
        [task_config, audio_qa_mode],
        [text_llm_name],
    ).then(
        update_llm_params_visibility,
        [task_config, audio_qa_mode],
        [llm_params_accordion],
    )
    load_prompt_btn.click(refresh_system_prompts, [], [prompt_radio]).then(
        lambda: gr.update(visible=True), None, [prompt_panel]
    )
    close_prompt_btn.click(lambda: gr.update(visible=False), None, [prompt_panel])
    prompt_radio.change(
        load_system_prompt_preview,
        [prompt_radio],
        [prompt_preview]
    )
    confirm_prompt_btn.click(
        load_system_prompt_confirm,
        [prompt_radio],
        [system_prompt, prompt_panel]
    )

    save_prompt_btn.click(save_system_prompt, [system_prompt], [prompt_radio])
    save_btn.click(store_history, [chatbot, system_prompt], [history_radio])
    load_history_btn.click(lambda: gr.update(visible=True), None, [history_panel])
    close_history_btn.click(lambda: gr.update(visible=False), None, [history_panel])
    history_radio.change(
        load_history_preview,
        [history_radio],
        [history_preview]
    )
    confirm_history_btn.click(
        load_history_confirm,
        [history_radio],
        [chatbot, system_prompt, history_panel]
    )
    
    chatbot.like(
        show_feedback, 
        inputs=[], 
        outputs=[additional_feedback, user_binary_feedback]
    )
    user_additional_feedback_submit.click(
        save_feedback,
        [user_binary_feedback, chatbot, system_prompt, context_html, llm_name, user_additional_feedback],
        [additional_feedback]
    )

    # --- Ingestion Events ---
    upload_btt.upload(upload_file_for_ingest, [upload_btt], [uploaded_doc_ing])
    run_ingestion.click(
        validate_ingestion_inputs,
        [index_name_ing, embed_name, upload_btt, chunk_length, percentile, retrievers_radio_ing],
        None
    ).success(
        lambda: gr.update(value="Ingesting...", visible=True), None, [ingestion_status]
    ).then(
        perform_ingest,
        [index_name_ing, chunk_length, percentile, embed_name, uploaded_doc_ing, splitting_strategy, retrievers_radio_ing],
        None
    ).then(
        lambda: gr.update(value="Success!", visible=True), None, [ingestion_status]
    )

    filter_column.change(process_filter_value_change, [filter_column, filter_value], [feedback_df, download_feedback, filter_value])
    filter_value.change(process_filter_value_change, [filter_column, filter_value], [feedback_df, download_feedback, filter_value])

app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    auth=authenticate,
    root_path=settings.ROOT_PATH,
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GRADIO_SERVER_PORT", 8080))
    uvicorn.run(
        "gradio_app.app:app",
        host="0.0.0.0",
        port=port,
        root_path=settings.ROOT_PATH,
        reload=True,
    )
