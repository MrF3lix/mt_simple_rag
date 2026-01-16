from openai import OpenAI
import logging

from generator import BaseGenerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class Generator(BaseGenerator):
    def __init__(self, cfg):
        logger.debug('Load Generator')
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

    def format_documents(self, document: dict) -> str:
        return f"<document><source>{document['id']}</source>{document['text']}</document>"
    
    def build_user_query(self, claim: str, documents: list[dict]):
        content = f"""
            Retrieved passages:
            {"\n".join(self.format_documents(p) for p in documents)}
            User Question:
            {claim}
            """
        return {
            "role": "user",
            "content": content
        }
    
    def generate(self, row):
        client = OpenAI(base_url=self.cfg.generator.base_url, api_key="dummy-key")

        question = row['question']
        context = row['context']

        request = self.build_user_query(question, context)

        response = client.chat.completions.create(
            model=self.cfg.generator.model,
            messages=[*self.history, request],
            temperature=self.cfg.generator.temperature,
            max_tokens=self.cfg.generator.max_tokens
        )

        return {
            **row,
            'generated_answer': response.choices[0].message.content
        }