import duckdb
import pandas as pd
from tqdm import tqdm

from .base_kb import KnowledgeBase
from index import DenseIndex, SparseIndex

class CatechismKnowledgeBase(KnowledgeBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.index = DenseIndex(cfg) if cfg.index.type == 'dense' else SparseIndex(cfg)
        self.con = duckdb.connect(cfg.knowledge_base.target)

    def init_database(self):
        self.select_subset()

        self.con.sql("DROP TABLE IF EXISTS paragraph")

        self.con.execute(f"CREATE TABLE paragraph (global_id BIGINT, text VARCHAR);")
        self.con.execute(f"""
        INSERT INTO paragraph
        SELECT 
            num AS global_id,
            text
        FROM '{self.cfg.knowledge_base.source}'
        """)

    def select_subset(self):
        df = pd.read_json(self.cfg.documents.source, lines=False)
        if 'subset_size' in self.cfg.documents:
            df = df.sample(self.cfg.documents.subset_size)

        df['id'] = df['num']
        df['input'] = df['question']

        df[['id', 'input', 'answer', 'references']].to_json(self.cfg.documents.target, lines=True, orient='records')

    def init_index(self):
        total = self.con.execute("SELECT count(*) FROM paragraph;").fetchall()[0][0]
        cursor = self.con.execute("SELECT global_id, text FROM paragraph")

        pbar = tqdm(total=total)
        while True:
            rows = cursor.fetchmany(self.cfg.index.batch_size)
            if not rows:
                break

            self.index.add_paragraphs(rows)
            pbar.update(self.cfg.index.batch_size)

        self.index.save_index()
