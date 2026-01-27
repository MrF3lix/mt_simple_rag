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
- [X] Cleanup unused scripts
- [X] Add Generator to complete RAG Pipeline
- [X] Add 10_000 query index (>200k paragraphs, 2k pages)

- [X] Add Config files
- [X] Add API Endpoint (Query, Eval)
- [ ] Define Deployment (e.g. What is needed? Code, KB, Index, ...)
- [ ] Define How to package the RAG and Run it?
- [ ] Create larger index



Next Steps
- [X] Common Input File Structure (Usable For multiple Tasks)
- [ ] Fix unseen experiment
- [ ] Index Entire Wiki
- [X] Sparse Index => BM25
- [x] Hybrid Index => Uses Sparse and Dense and Reranks the results
- [X] Use Compendium Dataset

- [ ] Add Simple Statistics (Count per Label)
- [X] Cleanup Code to work with both datasets
- [ ] Select another task from the kilt benchmark => Natural Questions



- How to go from the retrieved index id to the paragraph and wiki page?
- What should the API for the RAG look like
  - Input: Query, Config (Retriever Strategy, K, etc.)
  - Returns: Relevant Documents, Generated Answer

## Process

1. Load kilt wikipedia from huggingface
2. Load dataset into a duckdb
3. Init Knowledge Base
   1. Load kilt dataset from huggingface
   2. Select subset of fever benchmark => QUERY DATASET
   3. Select all relevant wikipages from subset
   4. Initialize duckdb with relevant wikipages and paragraphs
   5. Run embedding model (batched) => KNOWLEDGE BASE
4. Test Knowledge Base
   1. Load Embedding Index
5. Run the LLM => vLLM Endpoint
6.  Run the RAG API endpoint => RAG Endpoint
7.  Run the Evaluation Pipeline



## TODO

- [X] Create Wiki Dense Index
- [X] Create Wiki Sparse Index
- [X] Run 01_dense
- [X] Run 02_oracle
- [X] Run 03_random
- [X] Run 04_similar
- [X] Run 05_sparse
- [X] Run 06_hybrid
- [X] Run 07_probabilistic
- [X] Create Compendium Dense Index
- [X] Create Compendium Sparse Index
- [X] Run 01_dense
- [X] Run 02_oracle
- [X] Run 03_random
- [X] Run 04_similar
- [X] Run 05_sparse
- [X] Run 06_hybrid
- [X] Run 07_probabilistic
- [ ] Create Wiki NQ Dense Index
- [ ] Create Wiki NQ Sparse Index