"""
Credit to Derek Thomas, derek@huggingface.co
"""

# import subprocess
# subprocess.run(["pip", "install", "--upgrade", "transformers[torch,sentencepiece]==4.34.1"])

import logging
from time import perf_counter

import gradio as gr
import markdown
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.ChatGptInteractor import num_tokens_from_messages
from gradio_app.backend.query_llm import generate_hf, generate_openai, construct_openai_messages
from gradio_app.backend.semantic_search import table, embedder

from settings import *

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the template environment with the templates directory
env = Environment(loader=FileSystemLoader('gradio_app/templates'))

# Load the templates directly from the environment
context_template = env.get_template('context_template.j2')
context_html_template = env.get_template('context_html_template.j2')

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


def bot(history, api_kind):
    top_k_rank = 5
    thresh_dist = 1.2
    history[-1][1] = ""
    query = history[-1][0]

    if not query:
        gr.Warning("Please submit a non-empty string as a prompt")
        raise ValueError("Empty string was submitted")

    logger.info('Retrieving documents...')
    # Retrieve documents relevant to query
    document_start = perf_counter()

    query_vec = embedder.embed(query)[0]
    documents = table.search(query_vec, vector_column_name=VECTOR_COLUMN_NAME).limit(top_k_rank).to_list()
    thresh_dist = max(thresh_dist, min(d['_distance'] for d in documents))
    documents = [d for d in documents if d['_distance'] <= thresh_dist]
    documents = [doc[TEXT_COLUMN_NAME] for doc in documents]

    document_time = perf_counter() - document_start
    logger.info(f'Finished Retrieving documents in {round(document_time, 2)} seconds...')

    while len(documents) != 0:
        context = context_template.render(documents=documents)
        documents_html = [markdown.markdown(d) for d in documents]
        context_html = context_html_template.render(documents=documents_html)
        messages = construct_openai_messages(context, history)
        num_tokens = num_tokens_from_messages(messages, LLM_NAME)
        if num_tokens + 512 < context_lengths[LLM_NAME]:
            break
        documents.pop()
    else:
        raise gr.Error('Model context length exceeded, reload the page')

    for part in generate_openai(messages):
        history[-1][1] += part
        yield history, context_html
    else:
        print('Finished generation stream.')


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
                height=600,
            )

            with gr.Row():
                input_textbox = gr.Textbox(
                    scale=3,
                    show_label=False,
                    placeholder="Enter text and press enter",
                    container=False,
                )
                txt_btn = gr.Button(value="Submit text", scale=1)

            api_kind = gr.Radio(choices=["HuggingFace", "OpenAI"], value="OpenAI", label='Backend')

            # Examples
            gr.Examples(examples, input_textbox)

        with gr.Column():
            context_html = gr.HTML()

    # Turn off interactivity while generating if you click
    txt_msg = txt_btn.click(
        add_text, [chatbot, input_textbox], [chatbot, input_textbox], queue=False
    ).then(
        bot, [chatbot, api_kind], [chatbot, context_html]
    )

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)

    # Turn off interactivity while generating if you hit enter
    txt_msg = input_textbox.submit(add_text, [chatbot, input_textbox], [chatbot, input_textbox], queue=False).then(
        bot, [chatbot, api_kind], [chatbot, context_html])

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [input_textbox], queue=False)

demo.queue()
demo.launch(debug=True)
