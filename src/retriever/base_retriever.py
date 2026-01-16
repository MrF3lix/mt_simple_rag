from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class Paragraph:
    global_id: int
    document_id: int
    index: int
    text: str

@dataclass(frozen=True)
class TestCase:
    query: str
    references: list[Paragraph]
    retrieved: list[Paragraph]
 

class BaseRetriever(ABC):
    """Abstract Class for the retriever strategies"""

    @abstractmethod
    def retriev(self, case: TestCase) -> TestCase:
        """Executes the quantification method

        Parameters
        ---
        case : TestCase
            Test case containing the query, original paragraph references.

        Returns
        ---
        TestCase
            Test case with a retrieved list of paragraphs set.
        
        """
        pass