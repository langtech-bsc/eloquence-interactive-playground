import logging
import lancedb
from sentence_transformers import SentenceTransformer

from settings import *


# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
embedder = SentenceTransformer(EMB_MODEL_NAME)

db = lancedb.connect(LANCEDB_DIRECTORY)
table = db.open_table(LANCEDB_TABLE_NAME)
