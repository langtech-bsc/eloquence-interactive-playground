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

from prep_scripts.lancedb_setup import run_ingest, supported_file_types

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
    run_ingest([os.path.join(GENERIC_UPLOAD, fp) for fp in file_paths], chunk_size, percentile, embed_name, index_name, splitting_strategy)
    gr.Info("Ingestion Done!")


def validate(index_name, embedder, uploaded_files, chunk_length, percentile):
    if index_name is None or len(index_name) == 0:
        raise gr.Error("Please fill the index name.")
    if embedder is None or len(embedder) == 0:
        raise gr.Error("Please select an embedder.")
    if uploaded_files is None or len(uploaded_files) == 0:
        raise gr.Error("Please select at least one file.")
    for uf in uploaded_files:
        uf = os.path.basename(uf)
        if not any([uf.endswith(suff) for suff in supported_file_types]):
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


with gr.Blocks(theme=gr.themes.Monochrome(), css=CSS) as demo:
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
                                         file_types=supported_file_types,
                                         file_count="multiple",
                                         scale=0)
            supported_extensions = ", ".join([f'*.{sft}' for sft in supported_file_types])
            supported = gr.HTML(f"<span class='description'>Supported extensions [{supported_extensions}]</span>")
        with gr.Row():
            run_ingestion = gr.Button("Run Ingestion", scale=0)
        

    upload_btt.upload(upload_file, [upload_btt], [uploaded_doc])
    run_ingestion.click(validate, [index_name, embed_name, upload_btt, chunk_length, percentile]
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

demo.launch(debug=True)