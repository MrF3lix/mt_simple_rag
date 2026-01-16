from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import numpy as np
import faiss
import duckdb

QUESTIONS = 'data/kilt_fever_train_small.jsonl'
MODEL = 'jinaai/jina-embeddings-v3'
INDEX = 'data/kilt_wiki_small.index'
SOURCE = 'data/kilt_wiki_small.duckdb'


df_q = pd.read_json(QUESTIONS, lines=True)
model = SentenceTransformer(MODEL, trust_remote_code=True)
index = faiss.read_index(INDEX)
con = duckdb.connect(SOURCE)


def retriev(item, index, con, k=5):
    task = "retrieval.query"
    query = model.encode(
        item['input'],
        task=task,
        prompt_name=task,
    )

    distances, indices = index.search(np.array([query]), k)

    # TODO: Why???
    indices = indices+1

    result = con.execute("""
        SELECT *
        FROM paragraph
        WHERE global_id IN ({})
        """.format(",".join(map(str, indices[0])))).df()

    result['d'] = distances[0]

    # print(distances) # TODO: Find out why the distance is always the same?
    return result


def index_contains_duplicated(con):
    res = con.execute("SELECT * FROM paragraph").df()
    res['identifier'] = res.apply(lambda row: f"{row['wikipedia_id']}_{row['index']}", axis=1)
    ids = res['identifier']
    
    return len(res[ids.isin(ids[ids.duplicated()])].sort_values("identifier")) > 0

def extract_wikipedia_link(row):

    item = row['output'][0]['provenance'][0]

    return {
        'wikipedia_id': item['wikipedia_id'],
        'section': item['section'],
        'paragraph_id': item['start_paragraph_id'],
        'start_paragraph_id': item['start_paragraph_id'],
        'end_paragraph_id': item['end_paragraph_id']
    }


print('Is Index Trained:            ', 'True' if index.is_trained else 'False')
print('Index Contains Duplicated:   ', 'True' if index_contains_duplicated(con) else 'False')

results = []
for i, row in tqdm(df_q.iterrows(), total=df_q.shape[0]):
    ground_truth = extract_wikipedia_link(row)
    retrieved_documents = retriev(row, index, con)

    results.append({
        'i': i,
        'id': row['id'],
        'input': row['input'],
        'retrieved': retrieved_documents,
        'correct_document': ground_truth['wikipedia_id'] in retrieved_documents['wikipedia_id'].to_list(),
        'correct_paragraph': ground_truth['paragraph_id'] in retrieved_documents['index'].to_list()
    })

results = pd.DataFrame(results)

print('Number of Test Queries:      ', len(results))
print('Correct Documents:           ', results['correct_document'].sum() / len(results))
print('Correct Paragraph:           ', results['correct_paragraph'].sum() / len(results))

results.to_json('results.json', orient='records')