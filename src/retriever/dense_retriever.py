from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import duckdb

from .base_retriever import BaseRetriever
from .query import Paragraph

class DenseRetriever(BaseRetriever):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embedder.model, trust_remote_code=True)
        self.index = faiss.read_index(cfg.index.name)
        self.con = duckdb.connect(cfg.knowledge_base.target)

    def retriev(self, query: str) -> list[Paragraph]:
        query = self.model.encode(
            query,
            task=self.cfg.embedder.query_task,
            prompt_name=self.cfg.embedder.query_task,
        )

        distances, indices = self.index.search(np.array([query]), self.cfg.retriever.k)

        # TODO: Why???
        indices = indices+1

        result = self.con.execute("""
            SELECT *
            FROM paragraph
            WHERE global_id IN ({})
            """.format(",".join(map(str, indices[0])))).df()

        result['d'] = distances[0]
        # print(distances) # TODO: Find out why the distance is always the same? is this related to the index type?

        result = result.to_dict(orient='records')

        return list(map(lambda r: Paragraph(
            document_id=r['wikipedia_id'],
            index=r['index'],
            text=r['text'],
        ), result))