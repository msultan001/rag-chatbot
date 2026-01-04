## Repo overview

- Purpose: Retrieval-Augmented Generation (RAG) pipeline for analyzing consumer complaints across five financial products. Core, runnable logic currently lives in `notebooks/`.
- Canonical notebooks:
  - `notebooks/eda_preprocessing.ipynb` — EDA, filtering to five products, cleaning, writes `data/filtered_complaints.csv`.
  - `notebooks/chunking_embedding_and_indexing.ipynb` — chunking, embeddings (SentenceTransformers `all-MiniLM-L6-v2`), FAISS index creation saved under `vectorstore/` (script writes both `vectorstore/` and `vector_store/` for compatibility).
  - `notebooks/rag_pipeline_and_evaluation.ipynb` — loads vector store with LangChain, builds RetrievalQA, runs evaluation queries.

## What an AI coding agent should know (concise)

- Data locations: raw CSV expected at `data/raw/complaints.csv`. Preprocessed file produced at `data/filtered_complaints.csv`. Vector artifacts saved to `vectorstore/` (and `vector_store/` for backward compatibility).
- Notebooks are the working source; `src/` now contains programmatic helpers mirroring notebooks:
  - `src/data_preprocessing.py` — loading, filtering, `clean_text()`, and `stratified_sample()` (10k–15k target).
  - `src/chunking_embedding.py` — chunking, embedding and FAISS index build/save.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly). Use this model to keep consistency with notebooks and evaluation.
- Chunking params: `chunk_size=300`, `chunk_overlap=50` (LangChain RecursiveCharacterTextSplitter). Preserve these unless reviewers request different granularity.
- Vector store format: FAISS index (`complaint_index.faiss`) + `metadata.pkl` and `chunks.pkl` in `vectorstore/` (script also writes `vector_store/`).

## Rubric-specific checks (Task 1 & Task 2)

- Task 1 (EDA & Preprocessing): notebooks implement required EDA and cleaning steps and write `data/filtered_complaints.csv`. Programmatic helper `src/data_preprocessing.py` added to reproduce and produce a stratified sample.
- Task 2 (Chunking & Vector Store): notebooks implement chunking and FAISS creation; `src/chunking_embedding.py` added to produce the FAISS index programmatically. Missing before: explicit stratified sampling step — now implemented in `src/data_preprocessing.py` producing `data/filtered_complaints_sampled.csv` for Task 2 input.

## Developer workflows & commands

1. Create a virtual environment and install deps:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Produce Task 1 artifacts (filtered CSV + stratified sample):

```bash
python -m src.data_preprocessing
```

3. Build vector store (from sampled file if present, otherwise full filtered CSV):

```bash
python -m src.chunking_embedding
```

4. Run notebooks interactively for exploratory work and evaluation.

## Project conventions / gotchas

- Notebooks are authoritative: when in doubt, mirror their logic into `src/` helpers before refactoring. Tests are currently absent.
- Vector store paths are inconsistent in notebooks (`vectorstore/` vs `notebooks/vector_store/`). Use top-level `vectorstore/` (script writes both `vectorstore/` and `vector_store/`).
- The evaluation notebook uses a lightweight HF generation pipeline (`distilgpt2`) for local evaluation only — do not treat this as a production model.

## Examples (patterns to follow)

- Text cleaning (see `notebooks/eda_preprocessing.ipynb` and `src/data_preprocessing.py`): lowercase, remove boilerplate, strip special chars.
- Chunking (see `notebooks/chunking_embedding_and_indexing.ipynb` and `src/chunking_embedding.py`): use `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)`.
- Persist vector store: save FAISS index and parallel `metadata.pkl` mapping chunk index → original complaint metadata.

## When to ask the repo owner

- Clarify location/format of raw complaints CSV if it is not `data/raw/complaints.csv`.
- Confirm whether the sampled size for Task 2 should be 10k or larger (script defaults to 10k and caps at 15k for safety).

If you want, I can now:

- open a PR that ports remaining notebook logic into `src/` and add minimal unit tests, or
- generate a short README with step-by-step reproduction instructions.
