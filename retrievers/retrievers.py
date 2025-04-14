import os
import json
import time
import tqdm

import lancedb
import pyarrow as pa
import pandas as pd
import numpy as np
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, BSHTMLLoader, TextLoader


from gradio_app.backend.embedders import EmbedderFactory
from settings import *

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

class LanceDBRetriever:
    def __init__(self, db, threshold) -> None:
        self.emb_cache = {}
        self.threshold = threshold
        self.db = db
        self._load_index_config()

    def __call__(self, index_name, query, top_k=5):
        if index_name not in self.index_config:
            self._load_index_config()
        if index_name not in self.index_config:
            embedding_type = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            embedding_type = self.index_config[index_name]
        if embedding_type in self.emb_cache:
            embedder = self.emb_cache[embedding_type]
        else:
            # gr.Info("Loading embedding model", embedding_type)
            embedder = EmbedderFactory.get_embedder(embedding_type)
            self.emb_cache[embedding_type] = embedder

        table = self.db.open_table(index_name)
        query_vec = embedder.embed_query(query)
        documents = table.search(query_vec, vector_column_name=VECTOR_COLUMN_NAME)
        documents = documents.limit(top_k).to_list()
        if self.threshold:
            thresh_dist = THRESHOLD_DISTANCES[embedding_type]
            thresh_dist = max(thresh_dist, min(d['_distance'] for d in documents))
            documents = [d for d in documents if d['_distance'] <= self.threshold]
        documents = [doc[TEXT_COLUMN_NAME] for doc in documents]
        return documents
    
    def create(self, file_paths, chunk_size, percentile, embed_name, table_name, splitting_strategy):
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
            splitter = SemanticChunker(embedder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=float(percentile))
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
        self.index_config[table_name] = embed_name
        self._save_index_config()
        
        print(f'Embedding: {time_embed}, Ingesting: {time_ingest}')

    def _load_index_config(self):
        self.index_config = {}
        if os.path.exists(INDEX_CONFIG_PATH):
            with open(INDEX_CONFIG_PATH, "rt") as fd:
                self.index_config = json.load(fd)
    
    def _save_index_config(self):
        with open(INDEX_CONFIG_PATH, "wt") as fd:
            json.dump(self.index_config, fd, indent=4)
