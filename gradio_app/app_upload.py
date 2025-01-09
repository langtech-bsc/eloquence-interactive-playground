"""
Credit to Derek Thomas, derek@huggingface.co
"""

# import subprocess
# subprocess.run(["pip", "install", "--upgrade", "transformers[torch,sentencepiece]==4.34.1"])

import logging
from time import perf_counter

import gradio as gr
import os
import shutil
import lancedb
from jinja2 import Environment, FileSystemLoader

from prep_scripts.lancedb_setup import run_ingest

from settings import *

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the template environment with the templates directory
env = Environment(loader=FileSystemLoader('gradio_app/templates'))

# Load the templates directly from the environment
context_template = env.get_template('context_template.j2')
context_html_template = env.get_template('context_html_template.j2')

db = lancedb.connect(LANCEDB_DIRECTORY)
FOLDER =  "./upload"

def upload_file(file):
    gr.Info("Uploading File")
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
    shutil.copy(file, FOLDER)
    gr.Info("File Uploaded")


def perform_ingest(table_name, chunk_size, embed_name, file):
    if table_name is None or len(table_name) == 0:
        raise gr.Error("Index name must not be empty!")
        return
    if file is None or len(file) == 0:
        raise gr.Error("You must uplaod a PDF file first")
        return
    gr.Info("Ingesting the documents")
    run_ingest(os.path.join(FOLDER, file), chunk_size, embed_name, table_name)
    gr.Info("Ingestion Done!")

with gr.Blocks(theme=gr.themes.Monochrome(), css=CSS) as demo:
    with gr.Blocks():
        with gr.Row():
            index_name = gr.Textbox(label="Index Name")
            chunk_size = gr.Number(label="Length of the chunks", value=500)
        with gr.Row():
            embed_name = gr.Radio(
                choices=EMBEDDERS,
                value=EMBED_NAME,
                label='Embedder',
            )
        with gr.Row():
            uploaded_doc = gr.Textbox(label="Uploaded Document", visible=False)
        with gr.Row():
            ingestion_in_progress = gr.Text(visible=False)
        with gr.Row():
            upload_btt = gr.UploadButton("Select Document")
            run_ingestion = gr.Button("Run Ingestion")
        

    upload_btt.upload(upload_file, upload_btt).then(
        lambda fn: gr.Textbox(label="Uploaded Document",
                              visible=True,
                              interactive=False,
                              value=fn.split("/")[-1]),
        [upload_btt],
        [uploaded_doc])
    run_ingestion.click(lambda: gr.Textbox(interactive=False,
                                           visible=True,
                                           label="Status",
                                           elem_id="status",
                                           value="Ingestion in progress..."),
                        [],
                        [ingestion_in_progress])\
    .then(perform_ingest, [index_name, chunk_size, embed_name, upload_btt])\
    .then(lambda: gr.Textbox(visible=False),
          [],
          [ingestion_in_progress])
demo.launch(debug=True)