from retriever import Query

from .base_judge import BaseJudge

class DefaultJudge(BaseJudge):
    
    def evaluate(self, query: Query) -> Query:
        query.retrieved_correct_document = self.retrieved_correct_document(query)
        query.retrieved_correct_paragraph = self.retrieved_correct_paragraph(query)
        query.is_answer_correct = self.generated_answer_correct(query)

        return query
    
    def generated_answer_correct(self, query):
        return query.answer == query.generated_answer
