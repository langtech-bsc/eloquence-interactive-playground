import logging
import lancedb

from gradio_app.backend.embedders import EmbedderFactory
from settings import *


# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
embedder = EmbedderFactory.get_embedder(EMBED_NAME)

db = lancedb.connect(LANCEDB_DIRECTORY)
table = db.open_table(LANCEDB_TABLE_NAME)
