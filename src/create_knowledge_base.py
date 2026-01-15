import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

WIKI = 'data/kilt_wiki_small.jsonl'
SOURCE = 'data/kilt_wiki_small.duckdb'
EMBEDDING_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDING_TASK = 'text-matching'
INDEX = 'data/kilt_wiki_small.index'
BATCH_ROWS = 512
DIM = 1024

def create_knowledge_base(con):
    df = pd.read_json(WIKI, orient='records', lines=True)

    df_wiki = df[['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text', 'categories']]
    df_wiki.head(1)

    con.sql("DROP TABLE IF EXISTS wiki")
    con.sql('CREATE TABLE wiki AS SELECT * FROM df_wiki')
    con.sql("INSERT INTO wiki SELECT * FROM df_wiki")

    con.sql("DROP TABLE IF EXISTS paragraph")
    con.sql("CREATE TABLE paragraph (wikipedia_id VARCHAR, wikipedia_title VARCHAR, paragraph_id INTEGER, text VARCHAR);")

    con.execute("""
    INSERT INTO paragraph
    SELECT
        wikipedia_id,
        wikipedia_title,
        idx AS paragraph_id,
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

    dim = embeddings.shape[1]

    vectors = embeddings.astype("float32")

    id_index.add_with_ids(
        vectors,
        np.array(ids, dtype="int64")
    )


def add_paragraph_embeddings(con):
    con = duckdb.connect(SOURCE)

    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    index = faiss.IndexFlatIP(DIM)
    id_index = faiss.IndexIDMap2(index)

    # Collect total number of paragraphs
    total = con.execute("SELECT count(*) FROM paragraph;").fetchall()[0][0]
    pbar = tqdm(total=total)
    cursor = con.execute("SELECT * FROM paragraph")
    while True:
        rows = cursor.fetchmany(BATCH_ROWS)
        if not rows:
            break

        embed(rows, model, id_index)

        pbar.update(BATCH_ROWS)

    
    faiss.write_index(index, INDEX)
    print('Done')


def run():
    con = duckdb.connect(SOURCE)

    # create_knowledge_base(con)
    add_paragraph_embeddings(con)

run()