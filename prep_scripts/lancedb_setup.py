import shutil
import traceback

import lancedb
import torch
import pyarrow as pa
import pandas as pd
from pathlib import Path
import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer

from markdown_to_text import *
from settings import *


shutil.rmtree(LANCEDB_DIRECTORY, ignore_errors=True)
db = lancedb.connect(LANCEDB_DIRECTORY)
batch_size = 32

model = SentenceTransformer(EMB_MODEL_NAME)
model.eval()

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

schema = pa.schema([
    pa.field(VECTOR_COLUMN_NAME, pa.list_(pa.float32(), emb_sizes[EMB_MODEL_NAME])),
    pa.field(TEXT_COLUMN_NAME, pa.string()),
    pa.field(DOCUMENT_PATH_COLUMN_NAME, pa.string()),
])
tbl = db.create_table(LANCEDB_TABLE_NAME, schema=schema, mode="overwrite")

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

    doc_header = ' / '.join(split_path(file_path)) + ':\n\n'
    with open(file, encoding='utf-8') as f:
        f = f.read()
        f = remove_comments(f)
        f = split_markdown(f)
        chunks.extend((doc_header + chunk, os.path.abspath(file)) for chunk in f)

from matplotlib import pyplot as plt
plt.hist([len(c) for c, d in chunks], bins=100)
plt.show()

for i in tqdm.tqdm(range(0, int(np.ceil(len(chunks) / batch_size)))):
    texts, doc_paths = [], []
    for text, doc_path in chunks[i * batch_size:(i + 1) * batch_size]:
        if len(text) > 0:
            texts.append(text)
            doc_paths.append(doc_path)

    encoded = model.encode(texts, normalize_embeddings=True, device=device)
    encoded = [list(vec) for vec in encoded]

    df = pd.DataFrame({
        VECTOR_COLUMN_NAME: encoded,
        TEXT_COLUMN_NAME: texts,
        DOCUMENT_PATH_COLUMN_NAME: doc_paths,
    })

    tbl.add(df)


# '''
# create ivf-pd index https://lancedb.github.io/lancedb/ann_indexes/
# with the size of the transformer docs, index is not really needed
# but we'll do it for demonstration purposes
# '''
# tbl.create_index(num_partitions=256, num_sub_vectors=96, vector_column_name=VECTOR_COLUMN_NAME)

