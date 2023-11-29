import shutil
import time

import lancedb
import openai
import pyarrow as pa
import pandas as pd
from pathlib import Path
import tqdm
import numpy as np

from gradio_app.backend.embedders import EmbedderFactory
from markdown_to_text import *
from settings import *


with open('data/openaikey.txt') as f:
    OPENAI_KEY = f.read().strip()
openai.api_key = OPENAI_KEY


# shutil.rmtree(LANCEDB_DIRECTORY, ignore_errors=True)
db = lancedb.connect(LANCEDB_DIRECTORY)
batch_size = 32

schema = pa.schema([
    pa.field(VECTOR_COLUMN_NAME, pa.list_(pa.float32(), emb_sizes[EMBED_NAME])),
    pa.field(TEXT_COLUMN_NAME, pa.string()),
    pa.field(DOCUMENT_PATH_COLUMN_NAME, pa.string()),
])
table_name = f'{LANCEDB_TABLE_NAME}_{CHUNK_POLICY}_{EMBED_NAME}'
tbl = db.create_table(table_name, schema=schema, mode="overwrite")

input_dir = Path(MARKDOWN_SOURCE_DIR)
files = list(input_dir.rglob("*"))

chunks = []
for file in files:
    if not os.path.isfile(file):
        continue

    file_path, file_ext = os.path.splitext(os.path.relpath(file, input_dir))
    if file_ext != '.md':
        print(f'Skipped {file_ext} extension: {file}')
        continue

    with open(file, encoding='utf-8') as f:
        f = f.read()
        f = remove_comments(f)
        if CHUNK_POLICY == "txt":
            f = md2txt_then_split(f)
        else:
            assert CHUNK_POLICY == "md"
            f = split_markdown(f)
        chunks.extend((chunk, os.path.abspath(file)) for chunk in f)

from matplotlib import pyplot as plt
plt.hist([len(c) for c, d in chunks], bins=100)
plt.title(table_name)
plt.show()

embedder = EmbedderFactory.get_embedder(EMBED_NAME)

time_embed, time_ingest = [], []
for i in tqdm.tqdm(range(0, int(np.ceil(len(chunks) / batch_size)))):
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
print(f'Embedding: {time_embed}, Ingesting: {time_ingest}')



