import duckdb
import numpy as np
from tqdm import tqdm
from faiss import IndexHNSWFlat, write_index, METRIC_INNER_PRODUCT

EMBEDDER_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDER_TASK = 'text-matching'
TARGET_DB = 'data/all.duckdb'

DIM = 1024

INDEX_FILE = "data/all.index"
BATCH_SIZE = 2000

index = IndexHNSWFlat(DIM, 32, METRIC_INNER_PRODUCT)

con = duckdb.connect(TARGET_DB, read_only=True)
total = con.execute("SELECT count(*) FROM paragraph WHERE has_vec = TRUE").fetchall()[0][0]
pbar = tqdm(total=total)

while True:
    batch = con.execute("""
        SELECT global_id, vec
        FROM paragraph
        WHERE has_vec = TRUE
        ORDER BY global_id
        LIMIT ?
    """, [BATCH_SIZE]).fetchall()

    if not batch:
        print('Done')
        break

    for global_id, vec in batch:
        emb = np.asarray([vec], dtype="float32")
        index.add(emb)
        pbar.update(1)

write_index(index, INDEX_FILE)