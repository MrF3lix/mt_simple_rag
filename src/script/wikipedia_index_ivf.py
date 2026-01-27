import duckdb
import numpy as np
from tqdm import tqdm
import faiss

INDEX_FILE = "data/test_ivf.index"
TARGET_DB = 'data/all.duckdb'
DIM = 1024
BATCH_SIZE = 1000
NLIST = 65536
NPROBE = 16
NUM_THREADS = 15
TRAINING_SIZE = 2_555_904

faiss.omp_set_num_threads(NUM_THREADS)

con = duckdb.connect(TARGET_DB, read_only=True)

total = con.execute("""
    SELECT count(*) FROM paragraph WHERE has_vec = TRUE
""").fetchone()[0]

quantizer = faiss.IndexFlatIP(DIM)
index = faiss.IndexIVFFlat(
    quantizer,
    DIM,
    NLIST,
    faiss.METRIC_INNER_PRODUCT
)

train_batch = con.execute("""
    SELECT vec FROM paragraph
    WHERE has_vec = TRUE
    LIMIT ?
""", [TRAINING_SIZE]).fetchall()

train_embeddings = np.asarray([x[0] for x in train_batch], dtype="float32")
index.train(train_embeddings)

index.reserve(total)

pbar = tqdm(total=total)
offset = 0

while True:
    batch = con.execute("""
        SELECT vec FROM paragraph
        WHERE has_vec = TRUE
        LIMIT ? OFFSET ?
    """, [BATCH_SIZE, offset]).fetchall()

    if not batch:
        break

    embeddings = np.asarray([x[0] for x in batch], dtype="float32")
    index.add(embeddings)

    offset += len(embeddings)
    pbar.update(len(embeddings))

index.nprobe = NPROBE
faiss.write_index(index, INDEX_FILE)
