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

        # TODO: Cleanup with common datastructure
        if 'dataset' in self.cfg.knowledge_base and self.cfg.knowledge_base.dataset == 'catechism':
            result = self.con.execute(f"""
                SELECT *
                FROM paragraph
                WHERE global_id IN ({','.join(['?']*len(reference_paragraphs))})
            """, reference_paragraphs).df()
        else:
            document_id = query.references[0].document_id

            result = self.con.execute(f"""
                SELECT *
                FROM paragraph
                WHERE wikipedia_id = '{document_id}' AND index IN ({','.join(['?']*len(reference_paragraphs))})
            """, reference_paragraphs).df()

        result['d'] = 0
        result = result.to_dict(orient='records')

        query.retrieved = list(map(lambda r: Paragraph(
            document_id=r['wikipedia_id'] if 'wikipedia_id' in r else r['global_id'],
            global_id=r['global_id'],
            index=r['index'] if 'index' in r else r['global_id'],
            text=r['text'],
        ), result))

        return query