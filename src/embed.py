from sentence_transformers import SentenceTransformer
import pandas as pd

SOURCE = 'data/kilt_wiki_small.jsonl'
MODEL = 'jinaai/jina-embeddings-v3'


# TODO: This won't work for a larger dataset!
def split_pages_to_paragraphs(source):
    df = pd.read_json(source, lines=True)
    df['paragraphs'] = df['text'].apply(lambda t: t['paragraph'])

    cols = ['kilt_id', 'wikipedia_id', 'wikipedia_title']
    idx = pd.MultiIndex.from_tuples(df[cols].values.tolist(), names=cols)
    df_p = pd.DataFrame(df['paragraphs'].tolist(), idx).stack().reset_index(cols, name='paragraph')
    df_p['paragraph_id'] = df_p.index

    return df_p

def embed():
    df_p = split_pages_to_paragraphs(SOURCE)

    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    task = "text-matching"
    embeddings = model.encode(
        df_p['paragraph'].to_list(),
        task=task,
        prompt_name=task,
    )


    print(embeddings)

embed()