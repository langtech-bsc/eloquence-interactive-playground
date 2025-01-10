import time
import json
import os

import lancedb
import pyarrow as pa
import pandas as pd
from pathlib import Path
import tqdm
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from gradio_app.backend.embedders import EmbedderFactory
# from markdown_to_text import *
from settings import *


#with open('data/openaikey.txt') as f:
 #   OPENAI_KEY = f.read().strip()
#openai.api_key = OPENAI_KEY


def run_ingest(file_path, chunk_size, embed_name, table_name):
    db = lancedb.connect(LANCEDB_DIRECTORY)
    batch_size = 32

    schema = pa.schema([
        pa.field(VECTOR_COLUMN_NAME, pa.list_(pa.float32(), EMBEDDING_SIZES[EMBED_NAME])),
        pa.field(TEXT_COLUMN_NAME, pa.string()),
        pa.field(DOCUMENT_PATH_COLUMN_NAME, pa.string()),
    ])
    tbl = db.create_table(table_name, schema=schema, mode="overwrite")

    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(pages)

    chunks = [(doc.page_content, f"{doc.metadata['source']}-{doc.metadata['page']}") for doc in chunked_documents]
    embedder = EmbedderFactory.get_embedder(embed_name)

    time_embed, time_ingest = [], []
    for i in tqdm.tqdm(range(0, int(np.ceil(len(chunks) / batch_size))), desc="Ingesting"):
        texts, doc_paths = [], []
        for text, doc_path in chunks[i * batch_size:(i + 1) * batch_size]:
            if len(text) > 0:
                texts.append(text)
                doc_paths.append(doc_path)

        t = time.perf_counter()
        encoded = embedder.embed(texts)
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