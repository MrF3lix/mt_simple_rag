import duckdb

SOURCE_DB = 'data/compendium/catechism_english.jsonl'
TARGET_DB = 'data/compendium/catechism.duckdb'

con = duckdb.connect(TARGET_DB)

con.sql("DROP TABLE IF EXISTS paragraph")

con.execute(f"CREATE TABLE paragraph (global_id BIGINT, text VARCHAR);")

con.execute(f"""
INSERT INTO paragraph
SELECT 
    num AS global_id,
    text
FROM '{SOURCE_DB}'
""")
