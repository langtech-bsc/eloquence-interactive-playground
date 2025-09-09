import requests
from settings import settings

class RetrieverClient:
    def __init__(self, endpoint):
        if endpoint == "public":
            endpoint = settings.RETRIEVER_ENDPOINT
        self.endpoint = endpoint
    
    def search(self, index_name, query, top_k=5):
        params = {
            "index_name": index_name,
            "query": query,
            "top_k": top_k
        }

        response = requests.get(f"{self.endpoint}/search", params=params)
        if response.status_code == 200:
            return response.json()["documents"]
        else:
            return []
    
    def add(self, text, metadata, index_name):
        params = {
            "index_name": index_name,
            "text": text,
            "metadata": metadata
        }

        response = requests.get(f"{self.endpoint}/add", params=params)
        return response.status_code == 200
    
    def create_vs(self, files_to_upload, chunk_size, percentile, embed_name, table_name, splitting_strategy):
        files_to_upload = [("files", (open(f, "rb"))) for f in files_to_upload]
        # Other form data
        data = {
            "chunk_size": chunk_size,
            "percentile": percentile,
            "embed_name": embed_name,
            "table_name": table_name,
            "splitting_strategy": splitting_strategy,
        }

        response = requests.post(f"{self.endpoint}/create", files=files_to_upload, data=data)
        return response.status_code == 200
    
    def list_vs(self):
        print(requests.get(f"{self.endpoint}/list_indices").json())
        try:
            return requests.get(f"{self.endpoint}/list_indices").json()["index_names"]
        except:
            return []
