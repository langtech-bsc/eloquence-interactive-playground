"""
Credit to Derek Thomas, derek@huggingface.co
"""

# import subprocess
# subprocess.run(["pip", "install", "--upgrade", "transformers[torch,sentencepiece]==4.34.1"])

import logging
import os
import json
import sqlite3

import gradio as gr
import markdown
import lancedb
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.query_llm import LLMHandler
from gradio_app.backend.retrievers import LanceDBRetriever
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


def authenticate(user, password):
    db_conn = sqlite3.connect(SQL_DB).cursor()
    result = db_conn.execute(f"SELECT username FROM users WHERE username='{user}' and password='{password}'").fetchone()
    return result and result[0] == user


# Examples
examples = [
    "What is the goal of Eloquence?",
    "Which countries are participating?",
    "What is Omilia's task in this project?",
    "What computational resources are available?",
    "Do you prefer cats or dogs?"
]


def toggle_sidebar(state):
    state = not state
    return gr.update(visible = state), state


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, "")]
    return history, gr.Textbox(value="", interactive=False)


def get_dynamic_fields():
    return _get_tables(), _get_configs(), _get_prompts()


def _get_tables():
    return gr.Radio(
        label="Index name",
        choices=[t for t in vector_store.table_names()],
    )


def _get_prompts():
    with open(os.path.join(PROMPTS_PATH), "rt") as fd:
        prompts = json.load(fd)
    return gr.Dropdown(
        label="Prompt",
        choices=[(p["name"], p["prompt"]) for p in prompts]
    )

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


def update_prompt(selected_prompt):
    return selected_prompt


def validate(text, llm, top_k, temp, top_p, index_name, system_prompt, task_config):
    if len(text) == 0:
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

def interact(history, llm, top_k, temp, top_p, index_name, system_prompt, task_config):
    history[-1][1] = ""
    query = history[-1][0]
    task_config = json.loads(task_config)
    with open(INDEX_CONFIG_PATH, "rt") as fd:
        index_config = json.load(fd)
    if not query:
        raise gr.Error("Empty string was submitted")

    logger.info('Retrieving documents...')
    documents = [""]
    if task_config["RAG"]:
        embed_name = index_config[index_name]
        documents = retriever(index_name, query, embed_name, top_k)

    documents_html = [markdown.markdown(d) for d in documents]
    context_html = context_html_template.render(documents=documents_html)
    for part in llm_handler(llm, system_prompt, history, documents, temperature=temp, top_p=top_p):
        history[-1][1] += part
        yield history, context_html,  gr.update(visible=task_config["RAG"] == True)

with gr.Blocks(theme=gr.themes.Monochrome(), css=CSS,) as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                avatar_images=('assets/user.jpeg',
                            'assets/eloq.png'),
                bubble_full_width=True,
                show_copy_button=True,
                # show_share_button=True,
                height=400,
                label="EloquenceBot",
                # autoscroll=True,
            )

            with gr.Row():
                input_textbox = gr.Textbox(
                    scale=3,
                    show_label=False,
                    placeholder="Enter text and press enter",
                    container=False,
                )
                txt_btn = gr.Button(value="Submit text", scale=1)
            with gr.Row():
                gr.Examples(examples, input_textbox)
        
        with gr.Column(visible=False) as rag_column:
            context_html = gr.HTML()
    
    with gr.Row():
        with gr.Column():
            top_k = gr.Number(
                value=5,
                label="Top K documents",
            )
            temp = gr.Number(
                value=1.0,
                label="Temperature",
            )
            top_p = gr.Number(
                value=0.95,
                label="Top p",
            )
            index_name = gr.Radio(
                label="Index name",
            )
            task_config = gr.Radio(
                label="Task configuration",
            )
            llm_name = gr.Radio(
                choices=[
                    "gpt-3.5-turbo",
                    "meta-llama/Meta-Llama-3-8B",
                ],
                value="gpt-3.5-turbo",
                label='LLM'
            )
        with gr.Column():
            system_prompt = gr.Textbox(
                value="Enter prompt...",
                label="System Prompt:"
            )
            selected_prompt = gr.Dropdown(
                choices=[]
            )

    demo.load(
        get_dynamic_fields, [], [index_name, task_config, selected_prompt]
    )

    selected_prompt.change(update_prompt, [selected_prompt], [system_prompt])
    # Turn off interactivity while generating if you click
    txt_msg = txt_btn.click(
        validate, [input_textbox, llm_name, top_k, temp, top_p, index_name, system_prompt, task_config], []
    ).success(
        add_text, [chatbot, input_textbox], [chatbot, input_textbox], queue=False
    ).then(
        interact, [chatbot, llm_name, top_k, temp, top_p, index_name, system_prompt, task_config], [chatbot, context_html, rag_column], api_name="llm"
    )

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)

    # Turn off interactivity while generating if you hit enter
    txt_msg = input_textbox.submit(
        validate, [input_textbox, llm_name, top_k, temp, top_p, index_name, system_prompt, task_config], []
    ).success(
        add_text, [chatbot, input_textbox], [chatbot, input_textbox], queue=False
    ).then(
        interact, [chatbot, llm_name, top_k, temp, top_p, index_name, system_prompt, task_config], [chatbot, context_html, rag_column]
    )

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)

demo.queue()
# demo.launch(debug=True)
demo.launch(debug=True, auth=authenticate)
