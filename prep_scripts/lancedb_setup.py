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

from settings import *


emb_sizes = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "thenlper/gte-large": 0
}

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

schema = pa.schema(
  [
      pa.field(VECTOR_COLUMN_NAME, pa.list_(pa.float32(), emb_sizes[EMB_MODEL_NAME])),
      pa.field(TEXT_COLUMN_NAME, pa.string())
  ])
tbl = db.create_table(LANCEDB_TABLE_NAME, schema=schema, mode="overwrite")

input_dir = Path(TEXT_CHUNKS_DIR)
files = list(input_dir.rglob("*"))

sentences = []
for file in files:
    with open(file, encoding='utf-8') as f:
        sentences.append(f.read())

for i in tqdm.tqdm(range(0, int(np.ceil(len(sentences) / batch_size)))):
    try:
        batch = [sent for sent in sentences[i * batch_size:(i + 1) * batch_size] if len(sent) > 0]
        encoded = model.encode(batch, normalize_embeddings=True, device=device)
        encoded = [list(vec) for vec in encoded]

        df = pd.DataFrame({
            VECTOR_COLUMN_NAME: encoded,
            TEXT_COLUMN_NAME: batch
        })

        tbl.add(df)

    except:
        print(f"batch {i} was skipped: {traceback.format_exc()}")


'''
create ivf-pd index https://lancedb.github.io/lancedb/ann_indexes/
with the size of the transformer docs, index is not really needed
but we'll do it for demonstration purposes
'''
# tbl.create_index(num_partitions=256, num_sub_vectors=96, vector_column_name=VECTOR_COLUMN_NAME)

