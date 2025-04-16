import requests


def get_task_handler(config, llm, retriver):
    if "local" in config["service"]:
        return LocalTaskHandler(llm_handler=llm, retriever=retriver, task_config=config)
    if "remote" in config["service"]:
        return RemoteTaskHandler(task_config=config)
    

class LocalTaskHandler:
    def __init__(self, llm_handler, retriever, task_config):
        self.llm_handler = llm_handler
        self.retriever = retriever
        self.task_config = task_config
        
    def __call__(self, llm_name, system_prompt, history, query, docs_k, index_name, **params):
        documents = [""]
        if self.task_config["RAG"]:
            documents = self.retriever.search(index_name, query, docs_k)
        for part in self.llm_handler(llm_name, system_prompt, history, documents, **params):
            yield part, documents


class RemoteHandlerClient:
    def __init__(self, endpoint, method="POST"):
        self.endpoint = endpoint
        self.method = method
    
    def __call__(self, payload):
        call_f = requests.get if self.method == "GET" else requests.post
        response = call_f(url=self.endpoint, json=payload)
        return response


class RemoteTaskHandler:
    def __init__(self, task_config):
        self.task_config = task_config
        endpoint = task_config["service"].split("-")[1]
        self.client = RemoteHandlerClient(endpoint, method="POST")
    
    def _construct_payload(self, **params):
        return {k: v for k, v in params.items()}
    
    def __call__(self, llm_name, system_prompt, history, query, docs_k, index_name, **params):
        payload = self._construct_payload(query=query)
        response = self.client(payload)
        if response.status_code == 200:
            response = response.json()
            yield response["text"], response["documents"]
        else:
            yield "Error processing response from the remote service.", []
