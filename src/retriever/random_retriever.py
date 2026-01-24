import duckdb

from .base_retriever import BaseRetriever
from .query import Paragraph, Query

class RandomRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.con = duckdb.connect(cfg.knowledge_base.target)

    def retriev(self, query: Query) -> Query:
        reference_documents = list(map(lambda p: p.document_id, query.references))


        if 'dataset' in self.cfg.knowledge_base and self.cfg.knowledge_base.dataset == 'catechism':
            if len(reference_documents) == 0:
                reference_documents.append(-1)

            result = self.con.execute(f"""
                SELECT *
                FROM paragraph
                WHERE global_id NOT IN ({','.join(['?']*len(reference_documents))})
                USING SAMPLE 5;
            """, reference_documents).df()
        else:
            result = self.con.execute(f"""
                SELECT *
                FROM paragraph
                WHERE wikipedia_id NOT IN ({','.join(['?']*len(reference_documents))})
                USING SAMPLE 5;
            """, reference_documents).df()

        result = result.to_dict(orient='records')

        query.retrieved = list(map(lambda r: Paragraph(
            document_id=r['wikipedia_id'] if 'wikipedia_id' in r else r['global_id'],
            global_id=r['global_id'],
            index=r['index'] if 'index' in r else r['global_id'],
            text=r['text'],
        ), result))

        return query
