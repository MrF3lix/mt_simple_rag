from abc import ABC, abstractmethod
from retriever import TestCase


class BaseGenerator(ABC):
    """Abstract Class for the generator strategies"""

    @abstractmethod
    def generate(self, case: TestCase) -> TestCase:
        """Executes the generate method

        Parameters
        ---
        case : TestCase
            Test case containing the query, collected paragraph references

        Returns
        ---
        TestCase
            Test case with a generated answer based on the collected paragraphs.
        
        """
        pass