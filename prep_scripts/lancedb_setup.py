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



from settings import *



#             assert CHUNK_POLICY == "md"
#             f = split_markdown(f)
#         chunks.extend((chunk, os.path.abspath(file)) for chunk in f)

# from matplotlib import pyplot as plt
# plt.hist([len(c) for c, d in chunks], bins=100)
# plt.title(table_name)
# plt.show()
