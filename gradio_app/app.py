"""
Credit to Derek Thomas, derek@huggingface.co
"""

# import subprocess
# subprocess.run(["pip", "install", "--upgrade", "transformers[torch,sentencepiece]==4.34.1"])

import logging
import os
import json
import sqlite3
import shutil
import datetime
import time
import re

import gradio as gr
import markdown
import lancedb
from transformers import pipeline
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.query_llm import LLMHandler
from gradio_app.backend.task_handlers import get_task_handler
from retrievers.retrievers import LanceDBRetriever
from settings import *

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the template environment with the templates directory
env = Environment(loader=FileSystemLoader('gradio_app/templates'))

# Load the templates directly from the environment
context_template = env.get_template('context_template.j2')
context_html_template = env.get_template('context_html_template.j2')

vector_store = lancedb.connect(LANCEDB_DIRECTORY)
retriever = LanceDBRetriever(vector_store, threshold=None)
llm_handler = LLMHandler()


model_id = "openai/whisper-tiny"  # update with your model id
pipe = pipeline("automatic-speech-recognition", model=model_id)
def authenticate(user, password):
    db_conn = sqlite3.connect(SQL_DB).cursor()
    result = db_conn.execute(f"SELECT username FROM users WHERE username='{user}' and password='{password}'").fetchone()
    return True
    return result and result[0] == user


# Examples
examples = [
    "What is the goal of Eloquence?",
    "Which countries are participating?",
    "What is Omilia's task in this project?",
    "What computational resources are available?",
    "give me all information about the GPUS",
    "Do you prefer cats or dogs?"
]

############# FOR VS Creation #########
def upload_file(file_paths):
    out = []
    for file_path in file_paths:
        gr.Info("Uploading File")
        os.makedirs(GENERIC_UPLOAD, exist_ok=True)
        shutil.copy(file_path, GENERIC_UPLOAD)
        out.append(os.path.basename(file_path))
        gr.Info(f"File '{out[-1]}' uploaded")

    return gr.update(value=", ".join(out))


def perform_ingest(index_name, chunk_size, percentile,  embed_name, file_paths, splitting_strategy):
    if file_paths is None or len(file_paths) == 0:
        raise gr.Error("You must uplaod at least one file first")
    gr.Info("Ingesting the documents")
    retriever.create(
        [os.path.join(GENERIC_UPLOAD, fp) for fp in file_paths],
        chunk_size,
        percentile,
        embed_name,
        index_name,
        splitting_strategy)
    gr.Info("Ingestion Done!")


def validate_vs(index_name, embedder, uploaded_files, chunk_length, percentile):
    if index_name is None or len(index_name) == 0:
        raise gr.Error("Please fill the index name.")
    if embedder is None or len(embedder) == 0:
        raise gr.Error("Please select an embedder.")
    if uploaded_files is None or len(uploaded_files) == 0:
        raise gr.Error("Please select at least one file.")
    for uf in uploaded_files:
        uf = os.path.basename(uf)
        if not any([uf.endswith(suff) for suff in SUPPORTED_FILE_TYPES]):
            raise gr.Error(f"File '{uf}': filetype not supported!")
    try:
        chunk_length = int(chunk_length)
        if not 0 < chunk_length < 1000:
            raise ValueError()
    except:
        raise gr.Error("'Chunk Length' must be an integer between '0' and '1000'")
    try:
        percentile = int(percentile)
        if not 0 < percentile < 100:
            raise ValueError()
    except:
        raise gr.Error("'Percentile' must be an integer between '0' and '100'")

######## FOR PLAYGROUND ##############

def replace_doc_links(text):
    def repl(match):
        doc_id = match.group(1)
        url = f"#{doc_id}"
        return f'<a href="{url}" onmouseover="document.getElementById(\'doc_{doc_id}\').style=\'border: 2px solid white;background:#f27618\'; display: block;" onmouseout="document.getElementById(\'doc_{doc_id}\').style=\'border: 1px solid white; background: none; display:none;\'" >[{doc_id}]</a>'
    
    rep = re.sub(r"\[doc ?(\d+)\]", repl, text)
    return rep


def toggle_sidebar(state):
    state = not state
    return gr.update(visible = state), state



def get_dynamic_fields(request: gr.Request, selected_logs):
    return _get_tables(), _get_configs(), _get_prompts(request.username), _get_historical_prompts(request.username)


def _get_tables():
    return gr.Radio(
        label="Index name",
        choices=[t for t in vector_store.table_names()],
    )


def _get_prompts(user):
    prompts_path = os.path.join(USER_WORKSPACES, user if user is not None else "anonymous", "prompts.json")
    prompts = []
    if os.path.exists(prompts_path):
        with open(prompts_path, "rt") as fd:
            prompts = json.load(fd)
    return gr.Dropdown(
        label="Prompt",
        choices=[(p["name"], p["prompt"]) for p in prompts]
    )


def save_prompt(request: gr.Request, prompt):
    prompts_path = os.path.join(USER_WORKSPACES, request.username if request.username is not None else "anonymous", "prompts.json")
    os.makedirs(os.path.dirname(prompts_path), exist_ok=True)
    prompts = []
    if os.path.exists(prompts_path):
        with open(prompts_path, "rt") as fd:
            prompts = json.load(fd)
    prompts.append({"name": f"{prompt[:15]}...", "prompt": prompt})
    with open(prompts_path, "wt") as fd:
        json.dump(prompts, fd, indent=4)
    gr.Info("Prompt saved successfully!")


def _get_configs():
    configs = []
    for fn in os.listdir(TASK_CONFIG_DIR):
        with open(os.path.join(TASK_CONFIG_DIR, fn), "rt") as fd:
            content = json.load(fd)
            configs.append((content["name"], json.dumps(content)))
    return gr.Radio(
        label="Task configuration",
        choices=configs,
    )


def transcribe(filepath):
    try:
        output = pipe(
            filepath,
            max_new_tokens=256,
            generate_kwargs={
                "task": "transcribe",
                "language": "english",
            },  # update with the language you've fine-tuned on
            chunk_length_s=30,
            batch_size=8,
        )
        return output["text"]
    except:
        return ""



def _get_historical_prompts(user):
    history_path = os.path.join(USER_WORKSPACES, user if user is not None else "anonymous", "history.json")
    logs = []
    if os.path.exists(history_path):
        with open(history_path, "rt") as fd:
            logs = json.load(fd)
    return gr.Dropdown(
        label="History",
        choices=[p["name"] for p in logs]
    )


def update_prompt(selected_prompt):
    return selected_prompt


def validate(text, audio, llm, top_k, temp, top_p, index_name, system_prompt, task_config):
    if len(text) == 0 and len(audio) == 0:
        raise gr.Error("Empty query")
    if llm not in LLM_CONTEXT_LENGHTS:
        raise gr.Error("Unknown LLM")
    
    def _check_float(val, bmin, bmax):
        try:    
            val = float(val)
        except:
            return False
        if not bmin <= val <= bmax:
            return False
        return True

    if not _check_float(top_k, 0, 10):
        raise gr.Error("Accepted values for K are integers in range [0, 10]")
    if not _check_float(temp, 0, 2):
        raise gr.Error("Accepted values for Temperature are decimals in range [0, 2]")
    if not _check_float(top_p, 0, 1):
        raise gr.Error("Accepted values for Top-p are floats in range [0, 1]")

    if not task_config:
        raise gr.Error("Task configuration isn't selected")
    task_config = json.loads(task_config)
    if task_config["RAG"] and not index_name:
        raise gr.Error("Index required and not selected")


def _get_history_from_file(user, history_name):
    history_path = os.path.join(USER_WORKSPACES, user if user is not None else "anonymous", "history.json")
    if os.path.exists(history_path):
        with open(history_path, "rt") as fd:
            logs = json.load(fd)
    history = [l for l in logs if l["name"] == history_name]
    return history


def load_history(request: gr.Request, history_name, orig_history, orig_prompt):
    history = _get_history_from_file(request.username, history_name)
    if len(history) > 0:
        return history[0]["history"], history[0]["prompt"]
    else:
        return orig_history, orig_prompt


def prepare_download_history(request: gr.Request, history_name):
    history = _get_history_from_file(request.username, history_name)
    if len(history) > 0:
        dwnl_path = os.path.join(USER_WORKSPACES, request.username if request.username is not None else "anonymous", "history_download.json")
        with open(dwnl_path, "wt") as fd:
            json.dump(history[0], fd, indent=4)
        return gr.update(interactive=True, value=dwnl_path)
    else:
        return gr.update()
    


def reset_space():
    return "", "", "", gr.update(visible=False)


def load_task(task_config):
    task_config = json.loads(task_config)
    if task_config["interface"] == "audio":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

def store_history(request:gr.Request, history, prompt):
    history_path = os.path.join(USER_WORKSPACES, request.username if request.username is not None else "anonymous", "history.json")
    if os.path.exists(history_path):
        with open(history_path, "rt") as fd:
            logs = json.load(fd)
    else:
        logs = []
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
    logs.append({"history": history, "prompt": prompt, "name": f"{datetime.date.today()}-{prompt[:10]}..."})
    with open(history_path, "wt") as fd:
        json.dump(logs, fd)
    gr.Info(f"History Saved for '{request.username}'")


def upload_run_data(request: gr.Request, upload_file, history, input_text, audio_input, llm_name, top_k, temp, top_p, max_tokens, index_name, system_prompt, task_config, progress=gr.Progress()):
    gr.Info("Uploading File")
    upload_folder = os.path.join(USER_WORKSPACES, request.username if request.username is not None else "anonymous", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    shutil.copy(upload_file, upload_folder)
    gr.Info("File Uploaded")
    gr.Info(upload_file)
    with open(os.path.join(upload_folder, upload_file), "rt") as fd:
        data = json.load(fd)
        processed_data = {}
        for num, (conv_id, user_turns) in progress.tqdm(enumerate(data.items()), total=len(data)):
            contexts = []
            for turn in user_turns:
                for (partial_history,
                     context_html, 
                     gr_rag_update,
                     gr_textbox,
                     docs) in interact(history, turn, audio_input, llm_name, top_k, temp, top_p, max_tokens, index_name, system_prompt, task_config):
                    yield (partial_history,
                           context_html,
                           gr_rag_update,
                           gr.update()
                           )
                contexts.append(docs)
            time.sleep(.5)
            processed_data[conv_id] = [{"user": exchage[0],
                                        "LLM": exchage[1],
                                        "retrieved": context} for exchage, context in zip(partial_history, contexts)]
            history = []
            gr.Info(f"Conversation {num+1}/{len(data)} [{conv_id}] finished!")
        dwnl_path = os.path.join(upload_folder, "processed_data.json")
        with open(dwnl_path, "wt") as fd:
            json.dump(
                {"config": {
                    "task": task_config,
                    "top_k": top_k,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "LLM": llm_name,
                    "System prompt": system_prompt
                },
                "transcript": processed_data}, fd, indent=4)

        yield ([],
                "",
                gr.update(visible=False),
                gr.update(interactive=True, value=dwnl_path),
                )


def interact(history, input_text, audio_input, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config):
    task_config = json.loads(task_config)
    task_handler = get_task_handler(task_config, llm_handler, retriever)
    history = [] if history is None else history
    history_user_entry = None
    if task_config["interface"] == "audio":
        query = audio_input
        history_user_entry = query
    else:
        history_user_entry = input_text
        query = input_text

    if not query:
        raise gr.Error("Empty string was submitted")

    logger.info('Retrieving documents...')

    history += [[history_user_entry, ""]]
    for part, documents in task_handler(llm_name,
                                        system_prompt,
                                        history,
                                        query,
                                        docs_k,
                                        index_name,
                                        temperature=temp,
                                        top_p=top_p,
                                        max_tokens=max_tokens):
        history[-1][1] += part
        history[-1][1] = replace_doc_links(history[-1][1])
        documents_html = [markdown.markdown(d) for d in documents]
        context_html = context_html_template.render(documents=documents_html)
        yield (history,
               context_html,
               gr.update(visible=len(documents) > 1),
               gr.Textbox(value="", interactive=False),
               documents_html,
               )

JS = """
function Scrolldown() {
let targetNode = document.querySelector('[aria-label="chatbot conversation"]');
// Options for the observer (which mutations to observe)
const config = { attributes: true, childList: true, subtree: true };

// Callback function to execute when mutations are observed
const callback = (mutationList, observer) => {
targetNode.scrollTop = targetNode.scrollHeight;
};

// Create an observer instance linked to the callback function
const observer = new MutationObserver(callback);

// Start observing the target node for configured mutations
observer.observe(targetNode, config);

}

"""

"""
function highlightElement(_id) {
    parent = document.getElementById(_id);
    const descendants = parent.querySelectorAll("*");
    descendants.forEach(el => {
        el.style.color = 'red';
    });
};
"""

def remove_html_tags_and_content(text):
    return re.sub(r'<[^>]*>.*?</[^>]*>', '', text, flags=re.DOTALL)


def save_feedback(request: gr.Request, x: gr.LikeData, chatbot, system_prompt, rag):
    message = {
        "user": request.username,
        "feedback": int(x.liked),
        "history": remove_html_tags_and_content(chatbot[x.index[0]][x.index[1]]),
        "system_prompt": system_prompt
    }
    path = os.path.join(USER_WORKSPACES, "feedback.json")
    if os.path.exists(path):
        with open(path, "rt") as fd:
            feedback = json.load(fd)
    else:
        feedback = []
    feedback.append(message)
    
    with open(path, "wt") as fd:
        feedback = json.dump(feedback, fd, indent=4)
    gr.Info("Feedback saved")


with gr.Blocks(theme=gr.themes.Monochrome(), css=CSS, js=JS) as demo:
    with gr.Tab("Playground"):
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    avatar_images=('assets/user.jpeg',
                                'assets/eloq.png'),
                    bubble_full_width=True,
                    show_copy_button=False,
                    show_share_button=False,
                    height=400,
                    label="EloquenceBot",
                    sanitize_html=False
                )
                with gr.Row(equal_height=True):
                    with gr.Column(visible=True) as text_column:
                        input_textbox = gr.Textbox(
                            scale=3,
                            show_label=False,
                            container=False,
                        )
                    with gr.Column(visible=False) as audio_column:
                        audio_input = gr.Textbox(
                            visible=True
                        )
                        mic_transcribe = gr.Interface(
                            fn=transcribe,
                            inputs=gr.Audio(sources="microphone", type="filepath"),
                            outputs=audio_input,
                            allow_flagging="never",
                            live=True
                            # submit_btn=
                        )

                    txt_btn = gr.Button(value="Submit", scale=0)
                    save_btn = gr.Button(value="Save current state", scale=0)
                    clear_btn = gr.Button(value="Clear space", scale=0)

                with gr.Row():
                    gr.Examples(examples, input_textbox)

            with gr.Column(visible=False, scale=0) as rag_column:
                context_html = gr.HTML()

        with gr.Row():
            with gr.Column():
                docs_k = gr.Number(
                    value=5,
                    label="Top K documents",
                )
                temp = gr.Number(
                    value=1.0,
                    label="Temperature",
                )
                top_p = gr.Number(
                    value=0.95,
                    precision=2,
                    label="Top p",
                )
                max_tokens = gr.Number(
                    value=300,
                    label="Max tokens",
                )
                index_name = gr.Radio(
                    label="Index name",
                )
                task_config = gr.Radio(
                    label="Task configuration",
                )
                llm_name = gr.Radio(
                    choices=[
                        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
                        ("BSC (Salamandra-7B)", "bsc"),
                        # ("BSC (Salamandra-7B)", "bsc2"),
                        # ("BSC (EuroLLM-9B)", "bsc3"),
                    ],
                    value="gpt-3.5-turbo",
                    label='LLM'
                )
            with gr.Column():
                system_prompt = gr.Textbox(
                    value="",
                    label="System Prompt"
                )
                selected_prompt = gr.Dropdown(
                    choices=[]
                )
                save_prompt_btn = gr.Button(
                    value="Save prompt",
                    scale=0
                )
                with gr.Row():
                    selected_logs = gr.Dropdown(
                        choices=[]
                    )
                    select_log_btn = gr.Button(
                        value="Load history",
                        scale=0
                    )
                    download_btn = gr.DownloadButton(
                        label="Download history",
                        value=None,
                        scale=0,
                        interactive=False,
                    )
                with gr.Row():
                    upload_data_btt = gr.UploadButton("Upload & run from data...")
                    download_result_btn = gr.DownloadButton(
                        label="Download processed data",
                        value=None,
                        scale=0,
                        interactive=False,
                    )


        demo.load(
            get_dynamic_fields, [selected_logs], [index_name, task_config, selected_prompt, selected_logs]
        )

        selected_prompt.change(update_prompt, [selected_prompt], [system_prompt])
        selected_logs.change(prepare_download_history, [selected_logs], [download_btn])
        select_log_btn.click(load_history, [selected_logs, chatbot, system_prompt], [chatbot, system_prompt])
        task_config.change(load_task, [task_config], [text_column, audio_column])
        save_btn.click(store_history, [chatbot, system_prompt], [])
        clear_btn.click(reset_space, [], [chatbot, system_prompt, selected_prompt, rag_column])
        save_prompt_btn.click(save_prompt, [system_prompt], [])
        upload_data_btt.upload(
            validate, [gr.Textbox("dummy", visible=False), audio_input, llm_name, docs_k, temp, top_p, index_name, system_prompt, task_config], []
        ).success(
            upload_run_data,
            [upload_data_btt, chatbot, input_textbox, audio_input, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config],
            [chatbot, context_html, rag_column, download_result_btn]
        )
        chatbot.like(save_feedback, [chatbot, system_prompt, context_html], None)
        # .then(
        #         lambda fn: gr.Textbox(label="Uploaded Document",
        #                             visible=True,
        #                             interactive=False,
        #                             value=fn.split("/")[-1]),
        # Turn off interactivity while generating if you click
        txt_msg = txt_btn.click(
            validate, [input_textbox, audio_input, llm_name, docs_k, temp, top_p, index_name, system_prompt, task_config], []
        ).success(
            interact,
            [chatbot, input_textbox, audio_input, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config],
            [chatbot, context_html, rag_column, input_textbox],
            api_name="llm"
        )
        # Turn it back on
        txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)
        # Turn off interactivity while generating if you hit enter
        # txt_msg = input_textbox.submit(
        #     validate, [input_textbox, llm_name, top_k, temp, top_p, index_name, system_prompt, task_config], []
        # ).success(
        #     add_text, [chatbot, input_textbox], [chatbot, input_textbox], queue=False
        # ).then(
        #     interact, [chatbot, llm_name, top_k, temp, top_p, max_tokens, index_name, system_prompt, task_config], [chatbot, context_html, rag_column]
        # )
        # Turn it back on
        # txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)


    with gr.Tab("Ingestion"):
        with gr.Blocks():
            with gr.Row():
                index_name = gr.Textbox(label="Index Name")
                chunk_length = gr.Number(label="Chunk length (char-based)", value=500, scale=0)
                percentile = gr.Number(label="Percentile thr (sem-based)", value=95, precision=0, scale=0)
                splitting_strategy = gr.Radio(
                    label="Splitting strategy",
                    choices=[("By-Length (simple)", "simple"),
                            ("By-Length (recursive)", "recursive"),
                            ("Semantic", "semantic")],
                    value="recursive",
                    scale=1
                )
            with gr.Row():
                embed_name = gr.Radio(
                    choices=list(EMBEDDING_SIZES.keys()),
                    label="Embedder",
                )
            with gr.Row():
                uploaded_doc = gr.Textbox(label="Uploaded File(s)", interactive=False)
            with gr.Row():
                ingestion_in_progress = gr.Text(visible=False)
            with gr.Row():
                upload_btt = gr.UploadButton("Select file(s) to upload...",
                                            file_types=SUPPORTED_FILE_TYPES,
                                            file_count="multiple",
                                            scale=0)
                supported_extensions = ", ".join([f'*.{sft}' for sft in SUPPORTED_FILE_TYPES])
                supported = gr.HTML(f"<span class='description'>Supported extensions [{supported_extensions}]</span>")
            with gr.Row():
                run_ingestion = gr.Button("Run Ingestion", scale=0)
            

        upload_btt.upload(upload_file, [upload_btt], [uploaded_doc])
        run_ingestion.click(validate_vs, [index_name, embed_name, upload_btt, chunk_length, percentile]
                            ).success(
                                lambda: gr.Textbox(interactive=False,
                                                visible=True,
                                                label="Status",
                                                elem_id="status",
                                                value="Ingestion in progress..."),
                                [],
                                [ingestion_in_progress]
                            ).then(
                                perform_ingest,
                                [index_name, chunk_length, percentile, embed_name, upload_btt, splitting_strategy]
                            ).then(
                                lambda: gr.Textbox(visible=False),
                                [],
                                [ingestion_in_progress]
                            )
demo.queue()
# demo.launch(debug=True)
demo.launch(debug=True, auth=authenticate)

