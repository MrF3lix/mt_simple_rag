from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import numpy as np
import faiss
import duckdb

DATASET = 'small'
QUESTIONS = f'data/kilt_fever_train_{DATASET}.jsonl'
SOURCE = f'data/kilt_wiki_{DATASET}.duckdb'
EMBEDDING_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDING_TASK = 'retrieval.query'
INDEX = f'data/kilt_wiki_{DATASET}_2.index'

def retriev(item, index, con, model, k=4):
    query = model.encode(
        item['input'],
        task=EMBEDDING_TASK,
        prompt_name=EMBEDDING_TASK,
    )
    distances, indices = index.search(np.array([query]), k)
    # TODO: Why??? => Because the id_index is not saved
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
        'paragraph_id': item['start_paragraph_id'] + 1, # TODO: Necessary because the indices in the duckdb start at 1
        'start_paragraph_id': item['start_paragraph_id'] + 1,
        'end_paragraph_id': item['end_paragraph_id'] + 1
    }

def test():
    df_q = pd.read_json(QUESTIONS, lines=True)
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    index = faiss.read_index(INDEX)
    con = duckdb.connect(SOURCE)



    print('Is Index Trained:            ', 'True' if index.is_trained else 'False')
    print('Index Contains Duplicated:   ', 'True' if index_contains_duplicated(con) else 'False')

    results = []
    for i, row in tqdm(df_q.iterrows(), total=df_q.shape[0]):
        ground_truth = extract_wikipedia_link(row)
        retrieved_documents = retriev(row, index, con, model)

        results.append({
            'i': i,
            'id': row['id'],
            'input': row['input'],
            'ground_truth': ground_truth,
            'retrieved': retrieved_documents,
            'correct_document': ground_truth['wikipedia_id'] in retrieved_documents['wikipedia_id'].to_list(),
            'correct_paragraph': ground_truth['paragraph_id'] in retrieved_documents['index'].to_list()
        })

    results = pd.DataFrame(results)

    print('Number of Test Queries:      ', len(results))
    print('Correct Documents:           ', results['correct_document'].sum() / len(results))
    print('Correct Paragraph:           ', results['correct_paragraph'].sum() / len(results))

    results.to_json('results.json', orient='records')


test()