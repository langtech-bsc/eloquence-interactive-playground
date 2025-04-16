from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from retrievers.retrievers import LanceDBRetriever
import uvicorn
import lancedb
import shutil
import os
from settings import *


app = FastAPI()
vector_store = lancedb.connect(LANCEDB_DIRECTORY)
retriever = LanceDBRetriever(vector_store, threshold=None)


class SearchResult(BaseModel):
    documents: List[str]


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
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for file in files:
        file_location = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file_location)
    retriever.create(uploaded_files, chunk_size, percentile, embed_name, table_name, splitting_strategy)
    return "Success"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
