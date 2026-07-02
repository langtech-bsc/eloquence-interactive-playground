import os
import json
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

    def search_with_metadata(self, index_name, query, top_k=5):
        params = {
            "index_name": index_name,
            "query": query,
            "top_k": top_k
        }

        response = requests.get(f"{self.endpoint}/search", params=params)
        if response.status_code != 200:
            return [], []

        payload = response.json()
        documents = payload.get("documents", [])
        raw_metadata = payload.get("documents_metadata", [])
        metadata = []
        for item in raw_metadata:
            if isinstance(item, str):
                try:
                    metadata.append(json.loads(item))
                except json.JSONDecodeError:
                    metadata.append({"raw": item})
            else:
                metadata.append(item)

        while len(metadata) < len(documents):
            metadata.append(None)

        return documents, metadata
    
    def add(self, text, metadata, index_name):
        params = {
            "index_name": index_name,
            "text": text,
            "metadata": metadata
        }

        response = requests.get(f"{self.endpoint}/add", params=params)
        return response.status_code == 200

    def delete_vs(self, index_name):
        encoded_index_name = requests.utils.quote(index_name, safe="") # URL-encode the index name to handle special characters (space, "/"" , "#123", etc.)
        response = requests.delete(f"{self.endpoint}/index/{encoded_index_name}") # Make a DELETE request to the retriever server to delete the specified index
        if response.status_code != 200:
            raise RuntimeError(f"Retriever delete failed ({response.status_code}): {response.text}")
        return True
    
    def create_vs(self, files_to_upload, chunk_size, percentile, embed_name, table_name, splitting_strategy, append=False, metadata=None, turns=None):
        open_file_handles = []
        files_payload = []
        try:
            for file_path in files_to_upload:
                file_handle = open(file_path, "rb")
                open_file_handles.append(file_handle)
                files_payload.append(("files", (os.path.basename(file_path), file_handle)))

            data = {
                "chunk_size": chunk_size,
                "percentile": percentile,
                "embed_name": embed_name,
                "table_name": table_name,
                "splitting_strategy": splitting_strategy,
                "append": append,
            }
            if metadata is not None:
                data["metadata"] = json.dumps(metadata, ensure_ascii=False)
            if turns is not None:
                data["turns"] = json.dumps(turns, ensure_ascii=False)

            response = requests.post(f"{self.endpoint}/create", files=files_payload, data=data)
            if response.status_code != 200:
                raise RuntimeError(f"Retriever create failed ({response.status_code}): {response.text}")
            return True
        finally:
            for file_handle in open_file_handles:
                file_handle.close()
    
    def list_vs(self):
        print(requests.get(f"{self.endpoint}/list_indices").json())
        try:
            return requests.get(f"{self.endpoint}/list_indices").json()["index_names"]
        except:
            return []
