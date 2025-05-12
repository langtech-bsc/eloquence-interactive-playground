from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from retrievers.retrievers import LanceDBRetriever
import uvicorn
import lancedb
import shutil
import os
import argparse
from settings import settings


app = FastAPI()
vector_store = lancedb.connect(settings.LANCEDB_DIRECTORY)
retriever = LanceDBRetriever(vector_store, threshold=None)


class SearchResult(BaseModel):
    documents: List[str]


@app.get("/list_indices")
async def list_indices():
    return {"index_names": list(retriever.index_config.keys())}


@app.get("/search", response_model=SearchResult)
async def search_item(index_name: str, query: str, top_k: int = 5):
    results = retriever(index_name, query, int(top_k))
    response = SearchResult(documents=results)
    return response


@app.post("/create")
async def create_vs(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(...),
    percentile: float = Form(...),
    embed_name: str = Form(...),
    table_name: str = Form(...),
    splitting_strategy: str = Form(...)
):
    uploaded_files = []
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    for file in files:
        file_location = f"{settings.UPLOAD_DIR}/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file_location)
    retriever.create(uploaded_files, chunk_size, percentile, embed_name, table_name, splitting_strategy)
    return "Success"


@app.post("/add")
async def add_to_vs(text: str, metadata: str, index_name: str):
    retriever.add_single_chunk(text, metadata, index_name)
    return "Success"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.endpoint is None:
        endpoint = settings.RETRIEVER_ENDPOINT.replace("http://", "").split(":")
    else:
        endpoint = args.endpoint.replace("http://", "").split(":")
    uvicorn.run(app, host=endpoint[0], port=int(endpoint[1]))
