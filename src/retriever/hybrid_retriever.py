import copy
from .base_retriever import BaseRetriever
from .sparse_retriever import SparseRetriever
from .dense_retriever import DenseRetriever
from .query import Query

class HybridRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.sparse = SparseRetriever(cfg)
        self.dense = DenseRetriever(cfg)

    def retriev(self, query: Query) -> Query:

        query_sr = self.sparse.retriev(copy.deepcopy(query))
        query_dr = self.dense.retriev(copy.deepcopy(query))

        # TODO: Rerank the results and return the top K
        query.retrieved = query_sr.retrieved + query_dr.retrieved

        return query