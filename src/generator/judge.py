from openai import OpenAI
from retriever import Query, Paragraph

class Judge():
    def __init__(self, cfg):
        self.cfg = cfg

        self.history = [
            {
                "role": "user",
                "content": cfg.judge.system_prompt
            },
            {
                "role": "assistant",
                "content": "Hi, how can i help?"
            }
        ]

    def format_documents(self, document: Paragraph) -> str:
        return f"<document><source>{document.index}</source>{document.text}</document>"
    
    def build_user_query(self, query: Query):

        content = f"""
        ### Question:
        {query.input}
        ### Retrieved passages:
        {"\n".join(self.format_documents(p) for p in query.references)}
        ### Reference Answer:
        {query.answer}
        ### Generated Answer:
        {query.generated_answer}
        """

        return {
            "role": "user",
            "content": content
        }

    def eval(self, query: Query) -> bool:
        self.client = OpenAI(base_url=self.cfg.generator.base_url, api_key="dummy-key")
        request = self.build_user_query(query)

        response = self.client.chat.completions.create(
            model=self.cfg.generator.model,
            messages=[*self.history, request],
            temperature=self.cfg.generator.temperature,
            max_tokens=self.cfg.generator.max_tokens
        )

        eval_answer = response.choices[0].message.content
        eval_answer = eval_answer.strip().split('.')[0].lower()

        query.use_llm_judge = True
        query.is_answer_correct = True if eval_answer == "true" else False

        return query

