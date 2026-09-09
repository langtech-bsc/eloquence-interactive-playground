from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
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
    documents_metadata: List[Any] = Field(default_factory=list)


@app.get("/list_indices")
async def list_indices():
    retriever._load_index_config()
    return {
        "index_names": [
            index_name
            for index_name, embedder in retriever.index_config.items()
            if isinstance(embedder, str)
        ]
    }


@app.get("/search", response_model=SearchResult)
async def search_item(index_name: str, query: str, top_k: int = 5):
    try:
        results = retriever(index_name, query, int(top_k))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    documents = []
    documents_metadata = []
    for result in results:
        if isinstance(result, dict):
            documents.append(result.get("text", ""))
            documents_metadata.append(result.get("metadata"))
        else:
            documents.append(str(result))
            documents_metadata.append(None)
    response = SearchResult(documents=documents, documents_metadata=documents_metadata)
    return response


@app.post("/create")
async def create_vs(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(...),
    percentile: float = Form(...),
    embed_name: str = Form(...),
    table_name: str = Form(...),
    splitting_strategy: str = Form(...),
    append: bool = Form(False),
    metadata: str | None = Form(None),
    turns: str | None = Form(None),
):
    uploaded_files = []
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    for file in files:
        file_location = f"{settings.UPLOAD_DIR}/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file_location)
    try:
        retriever.create(
            uploaded_files,
            chunk_size,
            percentile,
            embed_name,
            table_name,
            splitting_strategy,
            append=append,
            metadata=metadata,
            turns=turns,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return "Success"


@app.post("/add")
async def add_to_vs(text: str, metadata: str, index_name: str):
    retriever.add_single_chunk(text, metadata, index_name)
    return "Success"


@app.delete("/index/{index_name}")
async def delete_index(index_name: str):
    try:
        retriever.delete_index(index_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
