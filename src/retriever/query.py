from dataclasses import dataclass

@dataclass(frozen=True)
class Paragraph:
    document_id: int
    index: int
    global_id: int = None
    text: str = None

class Query():
    id: str
    input: str
    answer: str
    references: list[Paragraph]
    retrieved: list[Paragraph]
    generated_answer: str = None

    def __init__(self, id, input, references, retrieved = [], answer = None):
        self.id = id
        self.input = input
        self.answer = answer
        self.references = references
        self.retrieved = retrieved

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