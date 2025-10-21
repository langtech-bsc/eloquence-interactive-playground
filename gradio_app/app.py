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
import base64
import tempfile
from copy import deepcopy
from typing import *

import gradio as gr
import markdown
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, Form, File
from pydantic import TypeAdapter
from fastapi.middleware.cors import CORSMiddleware


from gradio_app.backend.query_llm import LLMHandler
from gradio_app.backend.task_handlers import get_task_handler
from gradio_app.helpers import replace_doc_links
from gradio_app.messages import *

from retrievers.client import RetrieverClient
from settings import settings

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the template environment with the templates directory
env = Environment(loader=FileSystemLoader('gradio_app/templates'))

# Load the templates directly from the environment
context_template = env.get_template('context_template.j2')
context_html_template = env.get_template('context_html_template.j2')

dynamic_data = {
    "retriever_instance": None,
    "feedback_df": None,
    "audio_buffer": []
}
ALL_AVAILABLE_LLMS = list(settings.AVAILABLE_LLMS.keys()) + ["gpt-3.5-turbo"]

llm_handler = LLMHandler()


def authenticate(user, password):
    db_conn = sqlite3.connect(settings.SQL_DB).cursor()
    result = db_conn.execute(f"SELECT username FROM users WHERE username='{user}' and password='{password}'").fetchone()
    success = result and result[0] == user
    print(f"Login: {success}")
    return success


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
        os.makedirs(settings.GENERIC_UPLOAD, exist_ok=True)
        shutil.copy(file_path, settings.GENERIC_UPLOAD)
        out.append(os.path.basename(file_path))
        gr.Info(f"File '{out[-1]}' uploaded")

    return gr.update(value=", ".join(out))


def perform_ingest(index_name, chunk_size, percentile, embed_name, file_paths, splitting_strategy, retriever_address):
    if file_paths is None or len(file_paths) == 0:
        raise gr.Error("You must uplaod at least one file first")
    gr.Info("Ingesting the documents")
    retriever = RetrieverClient(endpoint=retriever_address)
    logger.info("retriever_address" + str(retriever_address))
    uploaded_files = [os.path.join(settings.GENERIC_UPLOAD, fp) for fp in file_paths]
    retriever.create_vs(
        uploaded_files,
        chunk_size,
        percentile,
        embed_name,
        index_name,
        splitting_strategy)
    shutil.rmtree(settings.GENERIC_UPLOAD, ignore_errors=True)
    gr.Info("Ingestion Done!")


def extract_docs_from_rendered_template(rendered):
    soup = BeautifulSoup(rendered, 'html.parser')
    text_list = [div.get_text(strip=True) for div in soup.select('.doc-box')]
    return text_list


def validate_vs(index_name, embedder, uploaded_files, chunk_length, percentile, retriever_addr):
    if index_name is None or len(index_name) == 0:
        raise gr.Error("Please fill the index name.")
    if retriever_addr is None or len(retriever_addr) == 0:
        raise gr.Error("Please choose a Vector Store instance.")
    if embedder is None or len(embedder) == 0:
        raise gr.Error("Please select an embedder.")
    if uploaded_files is None or len(uploaded_files) == 0:
        raise gr.Error("Please select at least one file.")
    for uf in uploaded_files:
        uf = os.path.basename(uf)
        if not any([uf.endswith(suff) for suff in settings.SUPPORTED_FILE_TYPES]):
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


def toggle_sidebar(state):
    state = not state
    return gr.update(visible = state), state



def get_dynamic_fields(request: gr.Request, selected_logs):
    feedback, avail_columns = _get_feedback()
    task_configs_radio, _ = _get_task_configs()
    retrievers_radio, _ = _get_retrievers(request.username)
    online_choices, _ = _get_online_models()
    return (
        task_configs_radio,
        _get_prompts(request.username),
        _get_historical_prompts(request.username),
        online_choices,
        retrievers_radio,
        retrievers_radio,
        feedback,
        avail_columns
    )


def change_retriever(selected_retr_endpoint):
    dynamic_data["retriever_instance"] = RetrieverClient(endpoint=selected_retr_endpoint)

    return gr.Radio(
        label="Index name",
        choices=[t for t in dynamic_data["retriever_instance"].list_vs()],
    )


def _load_feedback(force=False):
    feedback_fn = os.path.join(settings.USER_WORKSPACES, "user_feedback.json")
    data = []
    if not force and "feedback_df" in dynamic_data and dynamic_data["feedback_df"] is not None:
        return dynamic_data["feedback_df"]
    if os.path.exists(feedback_fn):
        with open(feedback_fn, "rt") as fd:
            data = json.load(fd)
    df = pd.DataFrame(data)
    dynamic_data["feedback_df"] = df
    return df


def _get_feedback():
    df = _load_feedback(force=True)
    return (
        gr.Dataframe(df, interactive=False),
        gr.Dropdown(
            label="Filter Columns",
            choices=["None"] + list(df.columns),
            value="None"
        )
    )


def _get_online_models():
    def _check_if_online(model_name):
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
        except:
            return False
        
    choices=[
        ("GPT", "gpt-3.5-turbo")
        ] + [(llm_entry.name, llm_entry.name) for llm_entry in settings.AVAILABLE_LLMS.values()]
    online_choices =  [choice for choice in choices
                       if _check_if_online(choice[1])]
    return gr.Radio(
        label="Available LLMs",
        choices=online_choices,
    ), [choice[1] for choice in online_choices]


def _get_retrievers(user):
    with open(settings.RETRIEVER_CONFIG_PATH) as fd:
        retrievers = json.load(fd)

    user_retriever_conf_path = os.path.join(settings.USER_WORKSPACES, user if user is not None else "anonymous", "retrievers.json")
    if os.path.exists(user_retriever_conf_path):
        with open(user_retriever_conf_path) as fd:
            retrievers.update(json.load(fd))
    
    return gr.Radio(
        label="Vector Store",
        choices=[(ret, addr) for ret, addr in retrievers.items()],
        
    ), retrievers


def _get_prompts(user):
    prompts_path = os.path.join(settings.USER_WORKSPACES, user if user is not None else "anonymous", "prompts.json")
    prompts = []
    if os.path.exists(prompts_path):
        with open(prompts_path, "rt") as fd:
            prompts = json.load(fd)
    return gr.Dropdown(
        label="Prompt",
        choices=[(p["name"], p["prompt"]) for p in prompts]
    )


def save_prompt(request: gr.Request, prompt):
    prompts_path = os.path.join(settings.USER_WORKSPACES, request.username if request.username is not None else "anonymous", "prompts.json")
    os.makedirs(os.path.dirname(prompts_path), exist_ok=True)
    prompts = []
    if os.path.exists(prompts_path):
        with open(prompts_path, "rt") as fd:
            prompts = json.load(fd)
    prompts.append({"name": f"{prompt[:15]}...", "prompt": prompt})
    with open(prompts_path, "wt") as fd:
        json.dump(prompts, fd, indent=4)
    gr.Info("Prompt saved successfully!")


def _get_task_configs():
    configs = []
    for fn in os.listdir(settings.TASK_CONFIG_DIR):
        with open(os.path.join(settings.TASK_CONFIG_DIR, fn), "rt") as fd:
            content = json.load(fd)
            configs.append((content["name"], json.dumps(content)))
    return gr.Radio(
        label="Task configuration",
        choices=configs,
    ), configs


def _get_historical_prompts(user):
    history_path = os.path.join(settings.USER_WORKSPACES, user if user is not None else "anonymous", "history.json")
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


def validate(text, llm, top_k, temp, top_p, index_name, task_config):
    if len(text) == 0:
        raise gr.Error("Empty query")
    if llm not in ALL_AVAILABLE_LLMS:
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
    history_path = os.path.join(settings.USER_WORKSPACES, user if user is not None else "anonymous", "history.json")
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
        dwnl_path = os.path.join(settings.USER_WORKSPACES, request.username if request.username is not None else "anonymous", "history_download.json")
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
        return gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False)
    else:
        return gr.update(visible=True), gr.update(visible=False), gr.update(interactive=True)

def store_history(request:gr.Request, history, prompt):
    history_path = os.path.join(settings.USER_WORKSPACES, request.username if request.username is not None else "anonymous", "history.json")
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


def upload_run_data(request: gr.Request, upload_file, history, input_text, llm_name, top_k, temp, top_p, max_tokens, index_name, system_prompt, task_config, progress=gr.Progress()):
    gr.Info("Uploading File")
    upload_folder = os.path.join(settings.USER_WORKSPACES, request.username if request.username is not None else "anonymous", "uploads")
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
                     docs) in interact(history, turn, llm_name, top_k, temp, top_p, max_tokens, index_name, system_prompt, task_config):
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


def interact(history, input_text, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config):
    task_config = json.loads(task_config)
    task_handler = get_task_handler(task_config, llm_handler, dynamic_data["retriever_instance"])
    history = [] if history is None else history
    history_user_entry = None
    audio_in = None
    history_user_entry = input_text
    query = input_text
    if task_config["interface"] == "audio":
        audio_in = deepcopy(dynamic_data["audio_buffer"])
        dynamic_data["audio_buffer"] = []
        query = "listen to  the audio"

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
                                        max_tokens=max_tokens,
                                        audio=audio_in):
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

async () => {
    let mediaRecorder = null;
    let socket = null;

    globalThis.Scrolldown = function() {
        let targetNode = document.querySelector('[aria-label="chatbot conversation"]');
        const config = { attributes: true, childList: true, subtree: true };

        const callback = (mutationList, observer) => {
        targetNode.scrollTop = targetNode.scrollHeight;
        };
        const observer = new MutationObserver(callback);
        observer.observe(targetNode, config);

    }
    globalThis.startStreaming = function() {
        const status = document.getElementById('recordstatus');
        status.innerText = "Requesting microphone access...";

        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                isStreaming = true;

                status.innerText = "Microphone recording... streaming audio (HTTP).";

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && isStreaming) {
                        const formData = new FormData();
                        formData.append("audio_chunk", event.data);

                        fetch("https://eloquence.lt.bsc.es/stream", {
                            method: "POST",
                            body: formData
                        }).catch(err => {
                            console.error("Upload error:", err);
                        });
                    }
                };

                mediaRecorder.start(250); // send every 250ms
            })
            .catch(err => {
                console.error("getUserMedia error:", err);
                status.innerText = "Error: " + err.name;
            });
    };

    globalThis.stopStreaming = function() {
        const status = document.getElementById('recordstatus');
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
            isStreaming = false;
            status.innerText = "Recording stopped.";
        }
    };

    globalThis.Scrolldown();
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


def process_filter_value_change(selected_col: str, selected_val: str):
    feedback_df = _load_feedback()
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


def remove_html_tags_and_content(text):
    return re.sub(r'<[^>]*>.*?</[^>]*>', '', text, flags=re.DOTALL)


def show_feedback(request: gr.Request, x: gr.LikeData):
    # This is currently not used:  x.index[0], x.index[1]
    return gr.update(visible=True), gr.update(value=str(x.liked))


def save_feedback(request: gr.Request, binary_feedback, chatbot, system_prompt, rag, model_name, custom_feedback):
    message = {
        "user": request.username,
        "feedback": binary_feedback,
        "custom_feedback": custom_feedback,
        "model": model_name,
        "system_prompt": system_prompt,
        "retrieved": extract_docs_from_rendered_template(rag),
        "generated_response": remove_html_tags_and_content(chatbot[-1][1]),
        "history": [remove_html_tags_and_content(msg) for turn in chatbot for msg in turn]
    }
    path = os.path.join(settings.USER_WORKSPACES, "user_feedback.json")
    if os.path.exists(path):
        with open(path, "rt") as fd:
            feedback = json.load(fd)
    else:
        feedback = []
    feedback.append(message)
    
    with open(path, "wt") as fd:
        feedback = json.dump(feedback, fd, indent=4)
    gr.Info("Feedback saved")
    return gr.update(visible=False)


with gr.Blocks(theme=gr.themes.Monochrome(), css=settings.CSS, js=JS) as demo:
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
                with gr.Row(visible=False) as additional_feedback:
                    user_binary_feedback = gr.Dropdown(label="Feedback", choices=["False", "True"])
                    user_additional_feedback = gr.Textbox(label="Additional Feedback")
                    user_additional_feedback_submit = gr.Button(value="Submit Feedback", scale=0)
                with gr.Row(equal_height=True):
                    with gr.Column(visible=True) as text_column:
                        input_textbox = gr.Textbox(
                            scale=3,
                            show_label=False,
                            container=False,
                        )
                    with gr.Column(visible=False) as audio_column:
                        hidden_submit_btn = gr.Button(visible=False, elem_id="trigger_audio_submit")
                        html = """
                        <button class="lg secondary  svelte-cmf5ev" onclick="startStreaming()">Start Streaming &#128266;</button>
                        <button class="lg secondary  svelte-cmf5ev" onclick="stopStreaming();document.getElementById('trigger_audio_submit').click();">Stop Streaming</button>
                        <p id="recordstatus">Recording stopped.</p>
                        """

                        audio_object = gr.HTML(html)

                    submit_btn = gr.Button(value="Submit", scale=0)
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
                retrievers_radio = gr.Radio(
                    label="Vector Store"
                )
                task_config = gr.Radio(
                    label="Task configuration",
                )
                llm_name = gr.Radio(
                    label='Available LLMs'
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


        selected_prompt.change(update_prompt, [selected_prompt], [system_prompt])
        selected_logs.change(prepare_download_history, [selected_logs], [download_btn])
        select_log_btn.click(load_history, [selected_logs, chatbot, system_prompt], [chatbot, system_prompt])
        task_config.change(load_task, [task_config], [text_column, audio_column, submit_btn])
        save_btn.click(store_history, [chatbot, system_prompt], [])
        clear_btn.click(reset_space, [], [chatbot, system_prompt, selected_prompt, rag_column])
        save_prompt_btn.click(save_prompt, [system_prompt], [])
        upload_data_btt.upload(
            validate, [gr.Textbox("dummy", visible=False), llm_name, docs_k, temp, top_p, index_name, task_config], []
        ).success(
            upload_run_data,
            [upload_data_btt, chatbot, input_textbox, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config],
            [chatbot, context_html, rag_column, download_result_btn]
        )
        retrievers_radio.change(change_retriever, [retrievers_radio], [index_name])
        chatbot.like(
            show_feedback, [], [additional_feedback, user_binary_feedback]
        )
        user_additional_feedback_submit.click(
            save_feedback, [user_binary_feedback, chatbot, system_prompt, context_html, llm_name, user_additional_feedback], [additional_feedback]
        )
        # .then(
        #         lambda fn: gr.Textbox(label="Uploaded Document",
        #                             visible=True,
        #                             interactive=False,
        #                             value=fn.split("/")[-1]),
        # Turn off interactivity while generating if you click
        txt_msg = submit_btn.click(
            validate, [input_textbox, llm_name, docs_k, temp, top_p, index_name, task_config], []
        ).success(
            interact,
            [chatbot, input_textbox, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config],
            [chatbot, context_html, rag_column, input_textbox],
            api_name="llm"
        )
        hidden_submit_btn.click(
            interact,
            [chatbot, input_textbox, llm_name, docs_k, temp, top_p, max_tokens, index_name, system_prompt, task_config],
            [chatbot, context_html, rag_column, input_textbox],
            api_name="llm"
        )

        # Turn it back on
        txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)

    with gr.Tab("Ingestion") as ing_tab:
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
                retrievers_radio_ing = gr.Radio(
                    label="Vector Store"
                )
            with gr.Row():
                embed_name = gr.Radio(
                    choices=list(settings.EMBEDDING_SIZES.keys()),
                    label="Embedder",
                )
            with gr.Row():
                uploaded_doc = gr.Textbox(label="Uploaded File(s)", interactive=False)
            with gr.Row():
                ingestion_in_progress = gr.Text(visible=False)
            with gr.Row():
                upload_btt = gr.UploadButton("Select file(s) to upload...",
                                            file_types=settings.SUPPORTED_FILE_TYPES,
                                            file_count="multiple",
                                            scale=0)
                supported_extensions = ", ".join([f'*.{sft}' for sft in settings.SUPPORTED_FILE_TYPES])
                supported = gr.HTML(f"<span class='description'>Supported extensions [{supported_extensions}]</span>")
            with gr.Row():
                run_ingestion = gr.Button("Run Ingestion", scale=0)
    
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
        get_dynamic_fields,
        [selected_logs],
        [task_config,
            selected_prompt,
            selected_logs,
            llm_name,
            retrievers_radio,
            retrievers_radio_ing,
            feedback_df,
            filter_column
            ]
    )
    filter_column.change(process_filter_value_change, [filter_column, filter_value], [feedback_df, download_feedback, filter_value])
    filter_value.change(process_filter_value_change, [filter_column, filter_value], [feedback_df, download_feedback, filter_value])
    upload_btt.upload(upload_file, [upload_btt], [uploaded_doc])
    run_ingestion.click(validate_vs, [index_name, embed_name, upload_btt, chunk_length, percentile, retrievers_radio_ing]
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
                            [index_name, chunk_length, percentile, embed_name, upload_btt, splitting_strategy, retrievers_radio_ing]
                        ).then(
                            lambda: gr.Textbox(visible=False),
                            [],
                            [ingestion_in_progress]
                        )
# demo.queue()
# demo.launch(debug=True)
# app = demo.launch(debug=True, auth=authenticate, prevent_thread_lock=True)
app = FastAPI()

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def query_llm_general(audio_file=None, **kwargs):
    if kwargs["llm_name"] not in ALL_AVAILABLE_LLMS:
        return ResponseQueryLLM(text="", documents=[], error="Unknown LLM")
    
    _, task_configs = _get_task_configs()
    task_configs = {k: json.loads(v) for k, v in task_configs}
    task_config = task_configs.get(kwargs["task_config"], settings.BASIC_CONFIG)
    retriever_address = kwargs["retriever_address"] if kwargs["retriever_address"] != "public" else settings.RETRIEVER_ENDPOINT
    task_handler = get_task_handler(task_config, llm_handler, RetrieverClient(endpoint=retriever_address))
    history = [] if kwargs["history"] is None else kwargs["history"]
    history_user_entry = None
    audio_in = None
    if task_config["interface"] == "audio":
        if audio_file:
            audio_in = await audio_file.read()
        else:
            audio_in = dynamic_data["audio_buffer"]
        query = "Describe the audio."
        history_user_entry = query
        dynamic_data["audio_buffer"] = []
    else:
        history_user_entry = kwargs["input_text"]
        query = kwargs["input_text"]

    if not query:
        raise gr.Error("Empty string was submitted")

    history += [[history_user_entry, ""]]

    received = []
    documents = []
    for part, documents in task_handler(kwargs["llm_name"],
                                        kwargs["system_prompt"],
                                        history,
                                        query,
                                        kwargs["docs_k"],
                                        kwargs["index_name"],
                                        temperature=kwargs["temp"],
                                        top_p=kwargs["top_p"],
                                        max_tokens=kwargs["max_tokens"],
                                        audio=audio_in):
        received += part
    return "".join(received), documents


@app.post("/stream")
async def upload_audio(audio_chunk: UploadFile):
    data = await audio_chunk.read()
    dynamic_data["audio_buffer"].extend(data)
    return {"status": "ok"}


@app.post("/query", response_model=ResponseQueryLLM)
async def query_llm(body: str = Form(...), audio_file: Optional[UploadFile] = File(None)):
    request = TypeAdapter(RequestQueryLLM).validate_json(body)
    response, documents = await query_llm_general(audio_file=audio_file, **request.__dict__)
    return ResponseQueryLLM(text=response, documents=documents)


@app.post("/batch_query", response_model=ResponseBatchQuery)
async def batch_query(body: str = Form(...), data_file: UploadFile = File(None)):
    batch_data = json.loads(await data_file.read())
    request = TypeAdapter(RequestBatchQuery).validate_json(body)
    processed_data = {}
    for conv, turns in batch_data.items():
        history = []
        for turn in turns:
            response, _ = await query_llm_general(audio_file=None, input_text=turn, history=deepcopy(history), **request.__dict__)
            history.append([turn, response])
        processed_data[conv] = history
    
    return ResponseBatchQuery(processed=processed_data)


@app.post("/ingest", response_model=ResponseIngest)
async def ingest(content_file: UploadFile, body: str = Form(...)):
    request = TypeAdapter(RequestIngest).validate_json(body)
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=content_file.filename) as tmp:
            shutil.copyfileobj(content_file.file, tmp)
            temp_file_path = tmp.name
            perform_ingest(request.index_name,
                        request.chunk_size,
                        request.percentile,
                        request.embed_name,
                        [temp_file_path],
                        request.splitting_strategy,
                        request.retriever_address)
        return ResponseIngest(status="success", msg="Document ingested successfully")
    except Exception as e:
        return ResponseIngest(status="error", msg=str(e))


@app.get("/list_vs", response_model=ResponseList)
async def list_available_stores(retriever_address: Optional[str] = "public"):
    retriever = RetrieverClient(endpoint=retriever_address)
    stores = retriever.list_vs()
    return ResponseList(available=stores)


@app.get("/list_llms", response_model=ResponseList)
def list_available_llms():
    _, online_models = _get_online_models()
    return ResponseList(available=online_models)


@app.get("/list_embedders", response_model=ResponseList)
def list_available_embedders():
    return ResponseList(available=list(settings.EMBEDDING_SIZES.keys()))


@app.get("/retrieval", response_model=ResponseList)
def get_retrieval(query: str,
                  index_name: str,
                  retriever_address: Optional[str] = "public",
                  top_k: Optional[int] = 5):
    retriever = RetrieverClient(endpoint=retriever_address)
    docs = retriever.search(index_name=index_name, query=query, top_k=top_k)
    return ResponseList(available=docs)


@app.get("/feedback", response_model=ResponseFeedback)
def download_feedback(filter_column: Optional[str] = None, filter_value: Optional[str] = None):
    feedback_df = _load_feedback(force=True)
    filt = ""
    if filter_column and filter_value:
        if filter_column in feedback_df.columns:
            filt = f"{filter_column}='{filter_value}'"
            feedback_df = feedback_df[feedback_df[filter_column] == filter_value]
    
    collected_feedback = feedback_df.to_dict(orient="records")
    return ResponseFeedback(filter=filt, feedback=collected_feedback)


app = gr.mount_gradio_app(app, demo, path="/", auth=authenticate)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gradio_app.app:app", host="0.0.0.0", port=int(os.environ.get("GRADIO_SERVER_PORT", "8080")), reload=True)
