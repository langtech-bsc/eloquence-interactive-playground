import os
import json
import tqdm

import lancedb
import pyarrow as pa
import pandas as pd
import numpy as np
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, BSHTMLLoader, TextLoader


from gradio_app.backend.embedders import EmbedderFactory
from settings import settings

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
    def __init__(self, db, threshold=None) -> None:
        self.emb_cache = {}
        self.threshold = threshold
        self.db = db
        self._load_index_config()

    def __call__(self, index_name, query, top_k=5):
        embedder = self._get_embedder(index_name)
        table = self.db.open_table(index_name)
        query_vec = embedder.embed_query(query)
        documents = table.search(query_vec, vector_column_name=settings.VECTOR_COLUMN_NAME)
        documents = documents.limit(top_k).to_list()
        if self.threshold:
            documents = [d for d in documents if d['_distance'] <= self.threshold]
        documents = [doc[settings.TEXT_COLUMN_NAME] for doc in documents]
        return documents

    def _get_embedder(self, index_name):
        if index_name not in self.index_config:
            self._load_index_config()
        if index_name not in self.index_config:
            embedding_type = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            embedding_type = self.index_config[index_name]
        if embedding_type in self.emb_cache:
            embedder = self.emb_cache[embedding_type]
        else:
            embedder = EmbedderFactory.get_embedder(embedding_type)
            self.emb_cache[embedding_type] = embedder
        return embedder
    
    def _add_batch_to_table(self, texts, metadata, embedder, tbl):
        encoded = embedder.embed_documents(texts)

        df = pd.DataFrame({
            settings.VECTOR_COLUMN_NAME: encoded,
            settings.TEXT_COLUMN_NAME: texts,
            settings.METADATA: metadata,
        })

        tbl.add(df)

    def create(self, file_paths, chunk_size, percentile, embed_name, table_name, splitting_strategy):
        db = lancedb.connect(settings.LANCEDB_DIRECTORY)
        batch_size = 128

        schema = pa.schema([
            pa.field(settings.VECTOR_COLUMN_NAME, pa.list_(pa.float32(), settings.EMBEDDING_SIZES[embed_name])),
            pa.field(settings.TEXT_COLUMN_NAME, pa.string()),
            pa.field(settings.METADATA, pa.string()),
        ])
        tbl = db.create_table(table_name, schema=schema, mode="overwrite")
        embedder = EmbedderFactory.get_embedder(embed_name)

        if splitting_strategy == "simple":
            splitter = CharacterTextSplitter(chunk_size=chunk_size)
        elif splitting_strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        else:
            splitter = SemanticChunker(embedder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=float(percentile))

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
                texts, metadata = [], []
                for text, md in chunks[i * batch_size:(i + 1) * batch_size]:
                    if len(text) > 0:
                        texts.append(text)
                        metadata.append(md)
                self._add_batch_to_table(texts, metadata, embedder, tbl)


        self.index_config[table_name] = embed_name
        self._save_index_config()
    

    
    def add_single_chunk(self, text: str, metadata: str, index_name: str):
        embedder = self._get_embedder(index_name)
        table = self.db.open_table(index_name)
        self._add_batch_to_table([text], [metadata], embedder, table)

    def _load_index_config(self):
        self.index_config = {}
        if os.path.exists(settings.INDEX_CONFIG_PATH):
            with open(settings.INDEX_CONFIG_PATH, "rt") as fd:
                self.index_config = json.load(fd)
    
    def _save_index_config(self):
        with open(settings.INDEX_CONFIG_PATH, "wt") as fd:
            json.dump(self.index_config, fd, indent=4)
