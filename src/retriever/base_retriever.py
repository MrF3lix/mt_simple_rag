from abc import ABC, abstractmethod
from .query import Query
 
class BaseRetriever(ABC):
    """Abstract Class for the retriever strategies"""

    @abstractmethod
    def retriev(self, query: Query) -> Query:
        """Executes the quantification method

        Parameters
        ---
        query : Query
            Uses the input of the query to search the index

        Returns
        ---
        Query
            Returns the query with a set list of retrieved paragraphs
        
        """
        pass