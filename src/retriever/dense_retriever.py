from sentence_transformers import SentenceTransformer
from retriever import BaseRetriever, TestCase
import numpy as np
import faiss
import duckdb

EMBEDDING_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDING_TASK = 'retrieval.query'
INDEX = 'data/kilt_wiki_small.index'
SOURCE = 'data/kilt_wiki_small.duckdb'
K = 5

class DenseRetriever(BaseRetriever):

    def __init__(self):
        super().__init__()

        self.model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        self.index = faiss.read_index(INDEX)
        self.con = duckdb.connect(SOURCE)

    def retriev(self, case: TestCase) -> TestCase:
        task = EMBEDDING_TASK
        query = self.model.encode(
            case.query,
            task=task,
            prompt_name=task,
        )

        distances, indices = self.index.search(np.array([query]), K)

        # TODO: Why???
        indices = indices+1

        result = self.con.execute("""
            SELECT *
            FROM paragraph
            WHERE global_id IN ({})
            """.format(",".join(map(str, indices[0])))).df()

        result['d'] = distances[0]
        # print(distances) # TODO: Find out why the distance is always the same? is this related to the index type?

        # TODO: Change to TestCase type
        return result