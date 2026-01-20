from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, ConfigDict

class Paragraph(BaseModel):
    document_id: int
    index: int
    global_id: Optional[int] = None
    text: Optional[str] = None

class Query(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    input: str
    answer: Optional[str] = None
    generated_answer: Optional[str] = None
    references: list[Paragraph] = []
    retrieved: list[Paragraph] = []

    # def __init__(self, id, input, references, retrieved = [], answer = None, generated_answer = None):
    #     super().__init__()
    #     self.id = id
    #     self.input = input
    #     self.answer = answer
    #     self.references = references
    #     self.retrieved = retrieved
    #     self.generated_answer = generated_answer

    def retrieved_correct_page(self):
        reference_page_ids = [r.document_id for r in self.references]
        retrieved_page_ids = [r.document_id for r in self.retrieved]

        return bool(set(reference_page_ids) & set(retrieved_page_ids))

    def retrieved_correct_paragraph(self):
        reference_paragraphs_ids = [r.index for r in self.references]
        retrieved_paragraphs_ids = [r.index for r in self.retrieved]

        return bool(set(reference_paragraphs_ids) & set(retrieved_paragraphs_ids))
    
    def generated_answer_correct(self):
        return self.answer == self.generated_answer


    def compute_result(self):
        return {
            'id': self.id,
            'input': self.input,
            'reference': self.references,
            'retrieved': self.retrieved,
            'answer': self.answer,
            'generated_answer': self.generated_answer,
            'correct_document': self.retrieved_correct_page(),
            'correct_paragraph': self.retrieved_correct_paragraph(),
            'correct_answer': self.generated_answer_correct(),
            'correct_query': True # TODO: Adjust once we have another dataset
        }