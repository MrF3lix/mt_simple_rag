import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

SOURCE = 'data/kilt_wiki_small.jsonl'
EMBEDDING_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDING_TASK = 'text-matching'
INDEX = 'data/kilt_wiki_small.index'


# TODO: This won't work for a larger dataset!
def split_pages_to_paragraphs(source):
    df = pd.read_json(source, lines=True)
    df['paragraphs'] = df['text'].apply(lambda t: t['paragraph'])

    cols = ['kilt_id', 'wikipedia_id', 'wikipedia_title']
    idx = pd.MultiIndex.from_tuples(df[cols].values.tolist(), names=cols)
    df_p = pd.DataFrame(df['paragraphs'].tolist(), idx).stack().reset_index(cols, name='paragraph')
    df_p['paragraph_id'] = df_p.index
    df_p = df_p.reset_index()

    return df_p

def embed():
    df_p = split_pages_to_paragraphs(SOURCE)

    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    embeddings = model.encode(
        df_p['paragraph'].to_list(),
        task=EMBEDDING_TASK,
        prompt_name=EMBEDDING_TASK,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX)

embed()