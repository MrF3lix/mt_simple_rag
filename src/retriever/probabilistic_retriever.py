from .base_retriever import BaseRetriever
from .oracle_retriever import OracleRetriever
from .random_retriever import RandomRetriever
from .query import Query

class ProbabilisticRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.oracle = OracleRetriever(cfg)
        self.random = RandomRetriever(cfg)

    def retriev(self, query: Query) -> Query:
        # TODO: Use the cfg.retriever.success_rate to determine wether to use the oracle or random retriever 
        return self.oracle.retriev(query)