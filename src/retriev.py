import json
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

    d, indices = index.search(np.array([query]), k)

    results = []
    for i in range(k):
        distance = d[0][i]
        index = indices[0][i]

        paragraph = df_p.iloc[index].to_dict()
        results.append({
            'kilt_id': paragraph['kilt_id'],
            'wikipedia_id': paragraph['wikipedia_id'],
            'wikipedia_title': paragraph['wikipedia_title'],
            'paragraph_id': paragraph['paragraph_id'],
            'd': distance.tolist()
        })

    return results

results = []
for i, row in df_q.iterrows():

    results.append({
        'i': i,
        'id': row['id'],
        'input': row['input'],
        'retrieved': retriev(row, index)
    })

with open('output.json', 'w') as f:
    json.dump(results, f)
