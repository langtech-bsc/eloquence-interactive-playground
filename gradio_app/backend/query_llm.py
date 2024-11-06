from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.ChatGptInteractor import *
from gradio_app.backend.HuggingfaceGenerator import HuggingfaceGenerator

env = Environment(loader=FileSystemLoader('gradio_app/templates'))
context_template = env.get_template('context_template.j2')
start_system_message = context_template.render(documents=[])


def construct_mistral_messages(context, history):
    messages = []
    for q, a in history:
        if len(a) == 0:  # the last message
            q = context + f'\n\nQuery:\n\n{q}'
        messages.append({
            "role": "user",
            "content": q,
        })
        if len(a) != 0:  # some of the previous LLM answers
            messages.append({
                "role": "assistant",
                "content": a,
            })
    return messages

def get_construct_openai_messages(system_prompt):
    def construct_openai_messages(context, history):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        for q, a in history:
            if len(a) == 0:  # the last message
                messages.append({
                    "role": "system",
                    "content": context,
                })
            messages.append({
                "role": "user",
                "content": q,
            })
            if len(a) != 0:  # some of the previous LLM answers
                messages.append({
                    "role": "assistant",
                    "content": a,
                })
        return messages
    return construct_openai_messages


def get_message_constructor(llm_name, system_prompt):
    if llm_name == 'gpt-3.5-turbo':
        return get_construct_openai_messages(system_prompt)
    if llm_name in ["meta-llama/Meta-Llama-3-8B",
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "tiiuae/falcon-180B-chat",
                    "GeneZC/MiniChat-3B"]:
        return construct_mistral_messages
    raise ValueError('Unknown LLM name')


def get_llm_generator(llm_name):
    if llm_name == 'gpt-3.5-turbo':
        cgi = ChatGptInteractor(
            model_name=llm_name, max_tokens=512, temperature=0, stream=True
        )
        return cgi.chat_completion
    print(llm_name)
    if llm_name in ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-Instruct-v0.1", "tiiuae/falcon-180B-chat"]:
        hfg = HuggingfaceGenerator(
            model_name=llm_name, temperature=0, max_new_tokens=512,
        )
        return hfg.generate

    if llm_name == "GeneZC/MiniChat-3B":
        hfg = HuggingfaceGenerator(
            model_name=llm_name, temperature=0, max_new_tokens=250, stream=False,
        )
        return hfg.generate
    raise ValueError('Unknown LLM name')



