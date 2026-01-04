## Intelligent Complaint Analysis

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for analyzing consumer complaints across five financial products.

Quick actions to reproduce rubric artifacts

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Produce Task 1 artifacts (EDA + cleaned CSV + stratified sample):

```bash
python -m src.data_preprocessing
```

This produces `data/filtered_complaints.csv` and `data/filtered_complaints_sampled.csv` (10k default).

3. Build vector store (Task 2):

```bash
python -m src.chunking_embedding
```

This builds embeddings with `sentence-transformers/all-MiniLM-L6-v2`, creates a FAISS index, and saves artifacts under `vectorstore/` (and `vector_store/` for compatibility): `complaint_index.faiss`, `metadata.pkl`, `chunks.pkl`.

Notes vs Rubric

- Task 1: `notebooks/eda_preprocessing.ipynb` contains exploratory plots and filtering logic; `src/data_preprocessing.py` provides programmatic helpers that implement the same filtering, cleaning, and stratified sampling required by the rubric.
- Task 2: `notebooks/chunking_embedding_and_indexing.ipynb` shows chunking and FAISS creation; `src/chunking_embedding.py` provides a CLI to create the vector store programmatically.

Repository structure (important files)

- `notebooks/` — canonical notebooks for EDA, chunking/embedding, and RAG evaluation.
- `src/data_preprocessing.py` — filtering, `clean_text()`, and `stratified_sample()`.
- `src/chunking_embedding.py` — chunking, embedding, FAISS index build/save.
- `vectorstore/` — produced FAISS index and metadata (or `vector_store/` for notebook compatibility).

If you'd like, I can add minimal unit tests that assert the presence and basic loadability of `data/filtered_complaints.csv` and `vectorstore/complaint_index.faiss`.
