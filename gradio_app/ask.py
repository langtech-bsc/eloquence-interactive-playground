import requests
import json
import re
from logzero import logger

def prepare_request(history: list[list[str]],
                    model_name: str,
                    embedder: str,
                    top_k_documents: int,
                    temperature: float,
                    top_p: float,
                    index_name: str,
                    sys_prompt: str,
                    task: str,
                    language:  str
                    ):
    return {
        "data": [
            history,
            model_name,
            embedder,
            top_k_documents,
            temperature,
            top_p,
            index_name,
            sys_prompt,
            task,
            language
        ]
    }

def get_event_id(payload, url):
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        return response.text, False
    event_id = response.json()
    return event_id["event_id"], True


if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 7860
    url = f"http://{HOST}:{PORT}/call/llm"

    history = [["What is the main goal of Eloquence?", ""]]
    model_name = "gpt-3.5-turbo"
    embedder = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_documents = 5
    temperature = 1
    top_p = 0.95
    index_name = "eloquence"
    system_prompt = "You should provide sound and complete answers to questions about the ELOQUENCE project."
    task = "RAG"
    payload = prepare_request(history=history,
                              model_name=model_name,
                              embedder=embedder,
                              top_k_documents=top_k_documents,
                              temperature=temperature,
                              top_p=top_p,
                              index_name=index_name,
                              sys_prompt=system_prompt,
                              task=task
                            )
    logger.info("Getting event ID")
    response, success = get_event_id(payload, url)
    if not success:
        print("Some error happened")
        print(response)
    else:
        logger.info("Reading Response")
        event_id = response
        response = requests.get(f"{url}/{event_id}")
        pattern = r'data:\s*(\[\[\[.*?\]\])'

        # Use the findall method to extract the data
        matches = re.findall(pattern, response.text.split("event: complete")[-1], re.DOTALL)

        # Print the extracted data
        if matches:
            data_extracted = matches[0]
            data_extracted = json.loads(data_extracted[1:])
            logger.info("Success!")
            print("Response:", data_extracted[0][1])
        else:
            print("No data found.")
