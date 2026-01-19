from dataclasses import dataclass

@dataclass(frozen=True)
class Paragraph:
    document_id: int
    index: int
    text: str = None

class Query():
    id: str
    input: str
    references: list[Paragraph]
    retrieved: list[Paragraph]

    def __init__(self, id, input, references, retrieved):
        self.id = id
        self.input = input
        self.references = references
        self.retrieved = retrieved

    def retrieved_correct_page(self):
        reference_page_ids = [r.document_id for r in self.references]
        retrieved_page_ids = [r.document_id for r in self.retrieved]

        return bool(set(reference_page_ids) & set(retrieved_page_ids))

    def retrieved_correct_paragraph(self):
        reference_paragraphs_ids = [r.document_id for r in self.references]
        retrieved_paragraphs_ids = [r.document_id for r in self.retrieved]

        return bool(set(reference_paragraphs_ids) & set(retrieved_paragraphs_ids))
    
    def compute_result(self):
        return {
            'id': self.id,
            'input': self.input,
            'reference': self.references,
            'retrieved': self.retrieved,
            'correct_document': self.retrieved_correct_page(),
            'correct_paragraph': self.retrieved_correct_paragraph(),
        }