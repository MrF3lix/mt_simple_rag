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
    use_llm_judge: bool = False
    is_answer_correct: Optional[bool] = None
    references: list[Paragraph] = []
    retrieved: list[Paragraph] = []

    def retrieved_correct_page(self):
        reference_page_ids = [r.document_id for r in self.references]
        retrieved_page_ids = [r.document_id for r in self.retrieved]

        return bool(set(reference_page_ids) & set(retrieved_page_ids))

    def retrieved_correct_paragraph(self):
        reference_paragraphs_ids = [r.index for r in self.references]
        retrieved_paragraphs_ids = [r.index for r in self.retrieved]

        return bool(set(reference_paragraphs_ids) & set(retrieved_paragraphs_ids))
    
    def generated_answer_correct(self):
        return self.is_answer_correct if self.use_llm_judge else self.answer == self.generated_answer


    def compute_result(self):
        return {
            'id': self.id,
            'input': self.input,
            'reference': list(map(lambda r: r.model_dump(), self.references)),
            'retrieved': list(map(lambda r: r.model_dump(), self.retrieved)),
            'answer': self.answer,
            'generated_answer': self.generated_answer,
            'correct_document': self.retrieved_correct_page(),
            'correct_paragraph': self.retrieved_correct_paragraph(),
            'correct_answer': self.generated_answer_correct(),
            'correct_query': True # TODO: Adjust once we have another dataset
        }