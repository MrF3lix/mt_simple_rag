import duckdb

from .base_retriever import BaseRetriever
from .query import Paragraph, Query

class OracleRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.con = duckdb.connect(cfg.knowledge_base.target)

    def retriev(self, query: Query) -> Query:
        reference_paragraphs = list(map(lambda p: p.index, query.references))

        # Reference is Empty so nothing can be retrieved.
        if len(reference_paragraphs) == 0:
            return query

        document_id = query.references[0].document_id
        result = self.con.execute(f"""
            SELECT *
            FROM paragraph
            WHERE document_id = '{document_id}' AND index IN ({','.join(['?']*len(reference_paragraphs))})
        """, reference_paragraphs).df()

        result['d'] = 0
        result = result.to_dict(orient='records')

        query.retrieved = self.results_to_paragraphs(result)

        return query