from abc import ABC, abstractmethod
from .query import Paragraph
 
class BaseRetriever(ABC):
    """Abstract Class for the retriever strategies"""

    @abstractmethod
    def retriev(self, query: str) -> list[Paragraph]:
        """Executes the quantification method

        Parameters
        ---
        query : str
            used to search the index

        Returns
        ---
        Paragraph
            list of retrieved paragraphs.
        
        """
        pass