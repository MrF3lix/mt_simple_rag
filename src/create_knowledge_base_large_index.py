import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

DATASET = 'small'
WIKI = f'data/kilt_wiki_{DATASET}.jsonl'
SOURCE = f'data/kilt_wiki_{DATASET}.duckdb'
EMBEDDING_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDING_TASK = 'text-matching'
INDEX = f'data/kilt_wiki_{DATASET}_2.index'

BATCH_ROWS = 10
DIM = 1024

def create_knowledge_base(con):
    df = pd.read_json(WIKI, orient='records', lines=True)

    df_wiki = df[['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text', 'categories']]
    df_wiki.head(1)

    con.sql("DROP TABLE IF EXISTS wiki")
    con.sql('CREATE TABLE wiki AS SELECT * FROM df_wiki')

    con.sql("DROP TABLE IF EXISTS paragraph")
    con.sql("CREATE TABLE paragraph (wikipedia_id VARCHAR, wikipedia_title VARCHAR, global_id BIGINT, index INTEGER, text VARCHAR);")

    con.sql("DROP SEQUENCE IF EXISTS paragraph_id")
    con.sql("CREATE SEQUENCE paragraph_id START 1;")

    con.execute("""
    INSERT INTO paragraph
    SELECT
        wikipedia_id,
        wikipedia_title,
        nextval('paragraph_id') as global_id,
        idx AS index,
        paragraph AS text
    FROM wiki, UNNEST(text.paragraph) WITH ORDINALITY AS t(paragraph, idx);
    """)

    return


def embed(batch, model, id_index):
    ids = [r[0] for r in batch]
    texts = [r[1] for r in batch]

    embeddings = model.encode(
        texts,
        task=EMBEDDING_TASK,
        prompt_name=EMBEDDING_TASK,
    )

    id_index.add(embeddings.astype("float32"))

    # id_index.add_with_ids(
    #     embeddings.astype("float32"),
    #     np.array(ids, dtype="int64")
    # )

def train_index(training_batch, model, id_index):
    embeddings = model.encode(
        training_batch,
        task=EMBEDDING_TASK,
        prompt_name=EMBEDDING_TASK,
    )
        
    train_vectors = embeddings.astype("float32")
    id_index.train(train_vectors)

def add_paragraph_embeddings(con):
    con = duckdb.connect(SOURCE)

    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    # TODO: Maybe switch to another index?
    # index = faiss.IndexFlatIP(DIM)

    # DIM = 1024
    # N_LISTS = 4096
    # M = 64
    # BITS = 8

    # quantizer = faiss.IndexFlatIP(DIM)
    # index = faiss.IndexIVFPQ(
    #     quantizer,
    #     DIM,
    #     N_LISTS,
    #     M,
    #     BITS
    # )

    index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)

    # print('Start Training')
    # training_rows = con.execute("SELECT global_id, text FROM paragraph LIMIT 4096").fetchall()
    # train_index(training_rows, model, index)

    # print('Finished Training')
    total = con.execute("SELECT count(*) FROM paragraph;").fetchall()[0][0]
    pbar = tqdm(total=total)
    cursor = con.execute("SELECT global_id, text FROM paragraph")
    while True:
        rows = cursor.fetchmany(BATCH_ROWS)
        if not rows:
            break

        embed(rows, model, index)
        pbar.update(BATCH_ROWS)

    faiss.write_index(index, INDEX)

def run():
    con = duckdb.connect(SOURCE)

    create_knowledge_base(con)
    add_paragraph_embeddings(con)

run()