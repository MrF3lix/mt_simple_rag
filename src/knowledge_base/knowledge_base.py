import duckdb
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
from . import Embedder

class KnowledgeBase():
    def __init__(self, cfg):
        self.cfg = cfg
        self.embedder = Embedder(cfg)
        self.con = duckdb.connect(cfg.knowledge_base.target)

    def init_database(self):
        self.con.sql("DROP TABLE IF EXISTS wiki")
        self.con.sql("DROP TABLE IF EXISTS paragraph")
        self.con.sql("DROP SEQUENCE IF EXISTS paragraph_id")

        self.init_wiki_table()

        self.con.sql("CREATE TABLE paragraph (wikipedia_id VARCHAR, wikipedia_title VARCHAR, global_id BIGINT, index INTEGER, text VARCHAR);")
        self.con.sql("CREATE SEQUENCE paragraph_id START 1;")

        self.con.execute("""
        INSERT INTO paragraph
        SELECT
            wikipedia_id,
            wikipedia_title,
            nextval('paragraph_id') as global_id,
            idx AS index,
            paragraph AS text
        FROM wiki, UNNEST(text.paragraph) WITH ORDINALITY AS t(paragraph, idx);
        """)

    def init_wiki_table(self):
        relevant_wiki_pages = self.select_subset()
        if 'use_entire_wiki' in self.cfg.documents.keys() and self.cfg.documents.use_entire_wiki == False:
            self.con.execute(f"""
                ATTACH '{self.cfg.knowledge_base.wiki_source}' AS src;
                
                CREATE TABLE wiki AS
                SELECT *
                FROM src.wiki
                WHERE wikipedia_id in ?;
            """,  [relevant_wiki_pages])
        else:
            self.con.execute(f"""
                ATTACH '{self.cfg.knowledge_base.wiki_source}' AS src;
                
                CREATE TABLE wiki AS
                SELECT *
                FROM src.wiki;
            """)

    def select_subset(self):
        kilt_fever = load_dataset("kilt_tasks", name="fever")
        allowed = ['SUPPORTS', 'REFUTES']

        train_clean = kilt_fever['train'].filter(lambda row: (row['output'][0]['answer'] in allowed and len(row['output'][0]['provenance']) > 0))

        if 'subset_size' in self.cfg.documents.keys():
            subset = train_clean.select(range(self.cfg.documents.subset_size))
        else:
            subset = train_clean

        relevant = subset.map(self.extract_wikipedia_link)
        relevant.to_json(self.cfg.documents.target)

        return relevant['wikipedia_id'][0:-1]

    def extract_wikipedia_link(sef, row):
        item = row['output'][0]['provenance'][0]

        return {
            'wikipedia_id': item['wikipedia_id']
        }

    def init_index(self):
        total = self.con.execute("SELECT count(*) FROM paragraph;").fetchall()[0][0]
        cursor = self.con.execute("SELECT global_id, text FROM paragraph")
        pbar = tqdm(total=total)
        while True:
            rows = cursor.fetchmany(self.cfg.index.batch_size)
            if not rows:
                break

            self.embedder.embed_paragraphs(rows)
            pbar.update(self.cfg.index.batch_size)

        self.embedder.save_index()
