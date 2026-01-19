# Simple RAG


## Questions

- [ ] How does the number of indexed paragraphs influence the performance?


## TODO

- [X] Create small version of KILT Fever
- [X] Extract all paragraphs from the smaller version of the dataset
- [X] Embedd all paragraphs
- [X] Create faiss index from embeddings
- [X] Test different indices => >1M Paragraphs
- [X] Fix Saving/Loading the ID Index then run it again on the Cluster => Needs a lot of memory!
- [ ] Cleanup unused scripts
- [ ] Add Generator to complete RAG Pipeline

- [X] Add Config files
- [ ] Add API Endpoint (Query, Eval)
- [ ] Define Deployment (e.g. What is needed? Code, KB, Index, ...)
- [ ] Define How to package the RAG and Run it?
- [ ] Create larger index


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
6. Run the LLM => vLLM Endpoint
7. Run the RAG API endpoint => RAG Endpoint
8. Run the Evaluation Pipeline