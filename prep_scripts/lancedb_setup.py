import time
import json
import os

import lancedb
import pyarrow as pa
import pandas as pd
from pathlib import Path
import tqdm
import numpy as np

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


from gradio_app.backend.embedders import EmbedderFactory
from settings import *

supported_file_types = ["pdf", "docx", "csv", "tsv", "html", "md", "txt"]

def get_doc_loader(file_path):
    extension = file_path.split(".")[-1]
    if extension == "pdf":
        return PyPDFLoader(file_path)
    elif extension == "docx":
        return Docx2txtLoader(file_path)
    elif extension == "csv":
        return CSVLoader(file_path, csv_args={"delimiter": ","})
    elif extension == "tsv":
        return CSVLoader(file_path, csv_args={"delimiter": "\t"})
    elif extension == "html":
        return BSHTMLLoader(file_path)
    elif extension in ["md", "txt"]:
        return TextLoader(file_path)
    else:
        raise NotImplementedError(f"Unknown extension {extension}")

def run_ingest(file_paths, chunk_size, embed_name, table_name, splitting_strategy):
    db = lancedb.connect(LANCEDB_DIRECTORY)
    batch_size = 128

    schema = pa.schema([
        pa.field(VECTOR_COLUMN_NAME, pa.list_(pa.float32(), EMBEDDING_SIZES[embed_name])),
        pa.field(TEXT_COLUMN_NAME, pa.string()),
        pa.field(DOCUMENT_PATH_COLUMN_NAME, pa.string()),
    ])
    tbl = db.create_table(table_name, schema=schema, mode="overwrite")
    embedder = EmbedderFactory.get_embedder(embed_name)

    if splitting_strategy == "simple":
        splitter = CharacterTextSplitter(chunk_size=chunk_size)
    elif splitting_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    else:
        splitter = SemanticChunker(embedder)
    time_embed, time_ingest = [], []

    for file_path in file_paths:
        loader = get_doc_loader(file_path)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)

        chunked_documents = splitter.split_documents(pages)
        try:
            chunks = [(doc.page_content, f"{doc.metadata['source']}-{doc.metadata['page']}") for doc in chunked_documents]
        except:
            chunks = [(doc.page_content, f"{doc.metadata}") for doc in chunked_documents]

        for i in tqdm.tqdm(range(0, int(np.ceil(len(chunks) / batch_size))), desc="Ingesting"):
            texts, doc_paths = [], []
            for text, doc_path in chunks[i * batch_size:(i + 1) * batch_size]:
                if len(text) > 0:
                    texts.append(text)
                    doc_paths.append(doc_path)

            t = time.perf_counter()
            encoded = embedder.embed_documents(texts)
            time_embed.append(time.perf_counter() - t)

            df = pd.DataFrame({
                VECTOR_COLUMN_NAME: encoded,
                TEXT_COLUMN_NAME: texts,
                DOCUMENT_PATH_COLUMN_NAME: doc_paths,
            })

            t = time.perf_counter()
            tbl.add(df)
            time_ingest.append(time.perf_counter() - t)


    time_embed = sum(time_embed)
    time_ingest = sum(time_ingest)
    cfg = {}
    if os.path.exists(INDEX_CONFIG_PATH):
        with open(INDEX_CONFIG_PATH, "rt") as fd:
            cfg = json.load(fd)
    cfg[table_name] = embed_name
    with open(INDEX_CONFIG_PATH, "wt") as fd:
        json.dump(cfg, fd, indent=4)
    print(f'Embedding: {time_embed}, Ingesting: {time_ingest}')



#             assert CHUNK_POLICY == "md"
#             f = split_markdown(f)
#         chunks.extend((chunk, os.path.abspath(file)) for chunk in f)

# from matplotlib import pyplot as plt
# plt.hist([len(c) for c, d in chunks], bins=100)
# plt.title(table_name)
# plt.show()