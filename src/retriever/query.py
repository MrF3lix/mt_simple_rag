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
    llm_judge_answer: Optional[str] = False
    is_answer_correct: bool = None
    retrieved_correct_document: Optional[bool] = None
    retrieved_correct_paragraph: Optional[bool] = None
    references: list[Paragraph] = []
    retrieved: list[Paragraph] = []

    def compute_result(self):
        return {
            'id': self.id,
            'input': self.input,
            'reference': list(map(lambda r: r.model_dump(), self.references)),
            'retrieved': list(map(lambda r: r.model_dump(), self.retrieved)),
            'answer': self.answer,
            'generated_answer': self.generated_answer,
            'use_llm_judge': self.use_llm_judge,
            'llm_judge_answer': self.llm_judge_answer,
            'correct_document': self.retrieved_correct_document,
            'correct_paragraph': self.retrieved_correct_paragraph,
            'correct_answer': self.is_answer_correct,
            'correct_query': True # TODO: Adjust once we have another dataset
        }