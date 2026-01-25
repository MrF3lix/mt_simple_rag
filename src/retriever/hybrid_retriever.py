from .base_retriever import BaseRetriever
from .sparse_retriever import SparseRetriever
from .dense_retriever import DenseRetriever
from .query import Query

class ProbabilisticRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.sparse = SparseRetriever(cfg)
        self.dense = DenseRetriever(cfg)

    def retriev(self, query: Query) -> Query:

        # TODO: Use Sparse Retriever
        # TODO: Use Dense Retriever

        # TODO: Rerank the results and return the top K

        return query