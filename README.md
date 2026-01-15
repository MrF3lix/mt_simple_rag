# Simple RAG


## TODO

- [ ] Create small version of KILT Fever
- [ ] Extract all paragraphs from the smaller version of the dataset
- [ ] Embedd all paragraphs
- [ ] Create faiss index from embeddings




- How to go from the retrieved index id to the paragraph and wiki page?
- What should the API for the RAG look like
  - Input: Query, Config (Retriever Strategy, K, etc.)
  - Returns: Relevant Documents, Generated Answer

## Process

1. Load kilt dataset from huggingface
2. Select subset of fever benchmark => QUERY DATASET
3. Select all relevant wikipages from subset
4. Initialize duckdb with relevant wikipages and paragraphs
5. Run embedding model (batched) => KNOWLEDGE BASE