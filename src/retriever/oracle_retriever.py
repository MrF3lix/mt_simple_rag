from retriever import BaseRetriever, TestCase
import duckdb

SOURCE = 'data/kilt_wiki_small.duckdb'

class OracleRetriever(BaseRetriever):
    def __init__(self):
        super().__init__()

        self.con = duckdb.connect(SOURCE)

    def retriev(self, case: TestCase) -> TestCase:
        reference_global_id = list(map(lambda p: p.global_id,case.references))

        query = f"""
            SELECT *
            FROM paragraph
            WHERE global_id IN ({','.join(['?']*len(reference_global_id))})
        """
        result = self.con.execute(query, reference_global_id).df()
        result['d'] = 0

        return result