import hashlib
import json
import os
import re

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import tqdm
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from gradio_app.backend.embedders import EmbedderFactory
from settings import settings


def normalize_index_name(index_name: str) -> str:
    """Return a LanceDB-safe table name for a user-facing index name."""
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", index_name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("Index name cannot be empty.")
    if normalized == index_name:
        return normalized
    digest = hashlib.sha1(index_name.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}_{digest}"


def get_doc_loader(file_path):
    extension = file_path.split(".")[-1]
    if extension == "pdf":
        return PyPDFLoader(file_path)
    if extension == "docx":
        return Docx2txtLoader(file_path)
    if extension == "csv":
        return CSVLoader(file_path, csv_args={"delimiter": ","})
    if extension == "tsv":
        return CSVLoader(file_path, csv_args={"delimiter": "\t"})
    if extension == "html":
        return BSHTMLLoader(file_path)
    if extension in ["md", "txt", "jsonl"]:
        return TextLoader(file_path)
    raise NotImplementedError(f"Unknown extension {extension}")


class LanceDBRetriever:
    def __init__(self, db, threshold=None) -> None:
        self.emb_cache = {}
        self.threshold = threshold
        self.db = db
        self._load_index_config()

    def __call__(self, index_name, query, top_k=5):
        index_name = normalize_index_name(index_name)
        embedder = self._get_embedder(index_name)
        table = self.db.open_table(index_name)
        query_vec = embedder.embed_query(query)

        # Select by semantic similarity, then restore the selected chunks to
        # their original ingestion order, as done by the SQA retriever.
        documents = (
            table.search(
                query_vec,
                vector_column_name=settings.VECTOR_COLUMN_NAME,
            )
            .with_row_id(True)
            .limit(int(top_k))
            .to_list()
        )
        if self.threshold is not None:
            documents = [
                document
                for document in documents
                if document["_distance"] <= self.threshold
            ]
        documents.sort(key=lambda document: document["_rowid"])
        return [
            {
                "text": document[settings.TEXT_COLUMN_NAME],
                "metadata": document.get(settings.METADATA),
            }
            for document in documents
        ]

    def _get_embedder(self, index_name):
        index_name = normalize_index_name(index_name)
        if index_name not in self.index_config:
            self._load_index_config()
        embedding_type = self.index_config.get(
            index_name,
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        if not isinstance(embedding_type, str):
            raise ValueError(
                f"Index '{index_name}' uses the removed multi-profile format. "
                "Re-ingest it as a single index."
            )
        if embedding_type not in self.emb_cache:
            self.emb_cache[embedding_type] = EmbedderFactory.get_embedder(
                embedding_type
            )
        return self.emb_cache[embedding_type]

    def _add_batch_to_table(self, texts, metadata, embedder, table):
        if not texts:
            return
        encoded = embedder.embed_documents(texts)
        frame = pd.DataFrame({
            settings.VECTOR_COLUMN_NAME: encoded,
            settings.TEXT_COLUMN_NAME: texts,
            settings.METADATA: metadata,
        })
        table.add(frame)

    @staticmethod
    def _turn_line(turn):
        speaker = str(turn.get("speaker") or "").strip()
        text = str(turn.get("text") or "").strip()
        if not text:
            return ""
        return f"{speaker}: {text}" if speaker else text

    def _chunks_from_turns(self, turns, chunk_size, source_metadata=None):
        max_chars = int(chunk_size or 500)
        snippets = []
        current_turns = []
        current_lines = []
        current_len = 0

        def flush_current():
            if not current_turns:
                return
            first_turn = current_turns[0]
            last_turn = current_turns[-1]
            snippet_metadata = {
                "start_time": first_turn.get("start_time"),
                "end_time": last_turn.get("end_time"),
                "turn_count": len(current_turns),
            }
            if source_metadata is not None:
                snippet_metadata["source_metadata"] = source_metadata
            snippets.append((
                "\n".join(current_lines).strip(),
                json.dumps(snippet_metadata, ensure_ascii=False),
            ))

        for turn in turns:
            if not isinstance(turn, dict):
                continue
            line = self._turn_line(turn)
            if not line:
                continue

            next_len = current_len + len(line) + (1 if current_lines else 0)
            if current_turns and next_len > max_chars:
                flush_current()
                current_turns = []
                current_lines = []
                current_len = 0

            current_turns.append(turn)
            current_lines.append(line)
            current_len += len(line) + (1 if current_len else 0)

        flush_current()
        return snippets

    def create(
        self,
        file_paths,
        chunk_size,
        percentile,
        embed_name,
        table_name,
        splitting_strategy,
        append=False,
        metadata=None,
        turns=None,
    ):
        table_name = normalize_index_name(table_name)
        db = lancedb.connect(settings.LANCEDB_DIRECTORY)
        batch_size = 128

        existing_table_names = {
            table.name if hasattr(table, "name") else str(table)
            for table in db.table_names()
        }
        if append and table_name in existing_table_names:
            table = db.open_table(table_name)
            if (
                table_name in self.index_config
                and self.index_config[table_name] != embed_name
            ):
                raise ValueError(
                    f"Embedder mismatch for existing index {table_name}"
                )
        else:
            schema = pa.schema([
                pa.field(
                    settings.VECTOR_COLUMN_NAME,
                    pa.list_(
                        pa.float32(),
                        settings.EMBEDDING_SIZES[embed_name],
                    ),
                ),
                pa.field(settings.TEXT_COLUMN_NAME, pa.string()),
                pa.field(settings.METADATA, pa.string()),
            ])
            mode = "create" if append else "overwrite"
            table = db.create_table(
                table_name,
                schema=schema,
                mode=mode,
            )
        embedder = EmbedderFactory.get_embedder(embed_name)

        if splitting_strategy == "simple":
            splitter = CharacterTextSplitter(chunk_size=chunk_size)
        elif splitting_strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        else:
            splitter = SemanticChunker(
                embedder,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=float(percentile),
            )

        if turns is not None:
            try:
                parsed_turns = (
                    json.loads(turns) if isinstance(turns, str) else turns
                )
                source_metadata = (
                    json.loads(metadata)
                    if isinstance(metadata, str) and metadata
                    else metadata
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid turn metadata JSON: {exc}"
                ) from exc

            chunks = self._chunks_from_turns(
                parsed_turns or [],
                chunk_size,
                source_metadata=source_metadata,
            )
            self._ingest_chunks(chunks, batch_size, embedder, table)
            self.index_config[table_name] = embed_name
            self._save_index_config()
            return

        for file_path in file_paths:
            loader = get_doc_loader(file_path)
            pages = list(loader.lazy_load())
            chunked_documents = splitter.split_documents(pages)
            if metadata is not None:
                chunks = [
                    (document.page_content, metadata)
                    for document in chunked_documents
                ]
            else:
                try:
                    chunks = [
                        (
                            document.page_content,
                            f"{document.metadata['source']}-"
                            f"{document.metadata['page']}",
                        )
                        for document in chunked_documents
                    ]
                except (KeyError, TypeError):
                    chunks = [
                        (document.page_content, str(document.metadata))
                        for document in chunked_documents
                    ]
            self._ingest_chunks(chunks, batch_size, embedder, table)

        self.index_config[table_name] = embed_name
        self._save_index_config()

    def _ingest_chunks(self, chunks, batch_size, embedder, table):
        batch_count = int(np.ceil(len(chunks) / batch_size))
        for batch_index in tqdm.tqdm(
            range(batch_count),
            desc="Ingesting",
        ):
            batch = chunks[
                batch_index * batch_size:(batch_index + 1) * batch_size
            ]
            texts = [text for text, _ in batch if text]
            metadata = [value for text, value in batch if text]
            self._add_batch_to_table(texts, metadata, embedder, table)

    def add_single_chunk(self, text: str, metadata: str, index_name: str):
        index_name = normalize_index_name(index_name)
        embedder = self._get_embedder(index_name)
        table = self.db.open_table(index_name)
        self._add_batch_to_table([text], [metadata], embedder, table)

    def delete_index(self, index_name: str):
        index_name = normalize_index_name(index_name)
        self._load_index_config()

        table_exists = True
        try:
            self.db.open_table(index_name)
        except Exception:
            table_exists = False

        index_in_config = index_name in self.index_config
        if not table_exists and not index_in_config:
            raise ValueError(f"Index '{index_name}' does not exist.")
        if table_exists:
            self.db.drop_table(index_name)
        if index_in_config:
            del self.index_config[index_name]
            self._save_index_config()

    def _load_index_config(self):
        self.index_config = {}
        if os.path.exists(settings.INDEX_CONFIG_PATH):
            with open(settings.INDEX_CONFIG_PATH, "rt") as config_file:
                self.index_config = json.load(config_file)

    def _save_index_config(self):
        with open(settings.INDEX_CONFIG_PATH, "wt") as config_file:
            json.dump(self.index_config, config_file, indent=4)
