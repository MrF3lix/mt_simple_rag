
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

EMBEDDING_MODEL = 'jinaai/jina-embeddings-v3'
PARAGRAPHS = 'data/kilt_wiki_small_paragraphs.jsonl'
QUESTIONS = 'data/kilt_fever_train_small.jsonl'
INDEX = 'data/kilt_wiki_small.index'

model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
index = faiss.read_index(INDEX)

df_q = pd.read_json(QUESTIONS, lines=True)
df_p = pd.read_json(PARAGRAPHS, lines=True)

def retriev(item, index, k=5):

    task = "retrieval.query"
    query = model.encode(
        item['input'],
        task=task,
        prompt_name=task,
    )

    return index.search(np.array([query]), k)
    
retriev(df_q.iloc[1], index)
print('done')