import os
import json
import duckdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

EMBEDDER_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDER_TASK = 'text-matching'
TARGET_DB = 'data/all.duckdb'

BATCH_SIZE = 25

OUTPUT_FILE = "embeddings.jsonl"

def load_processed_ids():
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed.add(obj["paragraph_id"])
                except json.JSONDecodeError:
                    pass
    return processed

def append_embeddings(results):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for pid, emb in results:
            record = {
                "paragraph_id": pid,
                "embedding": emb
            }
        f.write(json.dumps(record) + "\n")
        f.flush()


con = duckdb.connect(TARGET_DB)
model = SentenceTransformer(EMBEDDER_MODEL, trust_remote_code=True)

processed_ids = load_processed_ids()

total = con.execute("SELECT count(*) FROM paragraph LIMIT 1000").fetchall()[0][0]
cursor = con.execute("SELECT global_id, text FROM paragraph LIMIT 1000")

pbar = tqdm(total=total)
while True:
    batch = cursor.fetchmany(BATCH_SIZE)
    if not batch:
        break

    batch = [p for p in batch if p[0] not in processed_ids]
    if not batch:
        continue

    ids = [r[0] for r in batch]
    texts = [r[1] for r in batch]

    embeddings = model.encode(
        texts,
        task=EMBEDDER_TASK,
        prompt_name=EMBEDDER_TASK,
    )

    results = zip(ids, embeddings)
    append_embeddings(results)

    for id in ids:
        processed_ids.add(id)

