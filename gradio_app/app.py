"""
Credit to Derek Thomas, derek@huggingface.co
"""

# import subprocess
# subprocess.run(["pip", "install", "--upgrade", "transformers[torch,sentencepiece]==4.34.1"])

import logging
from time import perf_counter

import gradio as gr
import markdown
import lancedb
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.ChatGptInteractor import num_tokens_from_messages
from gradio_app.backend.cross_encoder import rerank_with_cross_encoder
from gradio_app.backend.query_llm import *
from gradio_app.backend.embedders import EmbedderFactory

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

# Examples
examples = [
    'What is BERT?',
    'Tell me about GPT',
    'How to use accelerate in google colab?',
    'What is the capital of China?',
    'Why is the sky blue?',
]


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, "")]
    return history, gr.Textbox(value="", interactive=False)


def bot(history, llm, cross_enc, chunk, embed):
    history[-1][1] = ""
    query = history[-1][0]

    if not query:
        raise gr.Error("Empty string was submitted")

    logger.info('Retrieving documents...')
    gr.Info('Start documents retrieval ...')
    t = perf_counter()

    table_name = f'{LANCEDB_TABLE_NAME}_{chunk}_{embed}'
    table = db.open_table(table_name)

    embedder = EmbedderFactory.get_embedder(embed)

    query_vec = embedder.embed([query])[0]
    documents = table.search(query_vec, vector_column_name=VECTOR_COLUMN_NAME)
    top_k_rank = TOP_K_RANK if cross_enc is not None else TOP_K_RERANK
    documents = documents.limit(top_k_rank).to_list()
    thresh_dist = thresh_distances[embed]
    thresh_dist = max(thresh_dist, min(d['_distance'] for d in documents))
    documents = [d for d in documents if d['_distance'] <= thresh_dist]
    documents = [doc[TEXT_COLUMN_NAME] for doc in documents]

    t = perf_counter() - t
    logger.info(f'Finished Retrieving documents in {round(t, 2)} seconds...')

    logger.info('Reranking documents...')
    gr.Info('Start documents reranking ...')
    t = perf_counter()

    documents = rerank_with_cross_encoder(cross_enc, documents, query)

    t = perf_counter() - t
    logger.info(f'Finished Reranking documents in {round(t, 2)} seconds...')

    msg_constructor = get_message_constructor(llm)
    while len(documents) != 0:
        context = context_template.render(documents=documents)
        documents_html = [markdown.markdown(d) for d in documents]
        context_html = context_html_template.render(documents=documents_html)
        messages = msg_constructor(context, history)
        num_tokens = num_tokens_from_messages(messages, 'gpt-3.5-turbo')  # todo for HF, it is approximation
        if num_tokens + 512 < context_lengths[llm]:
            break
        documents.pop()
    else:
        raise gr.Error('Model context length exceeded, reload the page')

    llm_gen = get_llm_generator(llm)
    logger.info('Generating answer...')
    t = perf_counter()
    for part in llm_gen(messages):
        history[-1][1] += part
        yield history, context_html
    else:
        t = perf_counter() - t
        logger.info(f'Finished Generating answer in {round(t, 2)} seconds...')


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                avatar_images=('https://aui.atlassian.com/aui/8.8/docs/images/avatar-person.svg',
                               'https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg'),
                bubble_full_width=False,
                show_copy_button=True,
                show_share_button=True,
                height=500,
            )

            with gr.Row():
                input_textbox = gr.Textbox(
                    scale=3,
                    show_label=False,
                    placeholder="Enter text and press enter",
                    container=False,
                )
                txt_btn = gr.Button(value="Submit text", scale=1)

            chunk_name = gr.Radio(
                choices=[
                    "md",
                    "txt",
                ],
                value="md",
                label='Chunking policy'
            )

            embed_name = gr.Radio(
                choices=[
                    "text-embedding-ada-002",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ],
                value="text-embedding-ada-002",
                label='Embedder'
            )

            cross_enc_name = gr.Radio(
                choices=[
                    None,
                    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                    "cross-encoder/ms-marco-MiniLM-L-12-v2",
                ],
                value=None,
                label='Cross-Encoder'
            )

            llm_name = gr.Radio(
                choices=[
                    "gpt-3.5-turbo",
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "tiiuae/falcon-180B-chat",
                    # "GeneZC/MiniChat-3B",
                ],
                value="gpt-3.5-turbo",
                label='LLM'
            )

            # Examples
            gr.Examples(examples, input_textbox)

        with gr.Column():
            context_html = gr.HTML()

    # Turn off interactivity while generating if you click
    txt_msg = txt_btn.click(
        add_text, [chatbot, input_textbox], [chatbot, input_textbox], queue=False
    ).then(
        bot, [chatbot, llm_name, cross_enc_name, chunk_name, embed_name], [chatbot, context_html]
    )

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)

    # Turn off interactivity while generating if you hit enter
    txt_msg = input_textbox.submit(add_text, [chatbot, input_textbox], [chatbot, input_textbox], queue=False).then(
        bot, [chatbot, llm_name, cross_enc_name, chunk_name, embed_name], [chatbot, context_html])

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)

demo.queue()
demo.launch(debug=True)
