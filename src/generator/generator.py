from openai import OpenAI
import logging

from generator import BaseGenerator
from retriever import Query, Paragraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class Generator(BaseGenerator):
    def __init__(self, cfg):
        self.cfg = cfg

        self.history = [
            {
                "role": "user",
                "content": cfg.generator.system_prompt
            },
            {
                "role": "assistant",
                "content": "Hi, how can i help?"
            }
        ]

    def format_documents(self, paragraph: Paragraph) -> str:
        return f"<document><source>{paragraph.document_id}, {paragraph.index}</source>{paragraph.text}</document>"
    
    def build_user_query(self, input: str, documents: list[Paragraph]):
        content = f"""
            Retrieved passages:
            {"\n".join(self.format_documents(p) for p in documents)}
            User Question:
            {input}
            """
        return {
            "role": "user",
            "content": content
        }
    
    def generate(self, query: Query) -> Query:
        client = OpenAI(base_url=self.cfg.generator.base_url, api_key="dummy-key")

        request = self.build_user_query(query.input, query.retrieved)

        response = client.chat.completions.create(
            model=self.cfg.generator.model,
            messages=[*self.history, request],
            temperature=self.cfg.generator.temperature,
            max_tokens=self.cfg.generator.max_tokens
        )

        query.generated_answer = response.choices[0].message.content

        return query