"""Chunking, embedding and FAISS vector store helpers.

Provides programmatic equivalents of the notebook steps:
- chunk cleaned narratives with LangChain's RecursiveCharacterTextSplitter
- create embeddings with SentenceTransformers `all-MiniLM-L6-v2`
- build and persist a FAISS index plus metadata
"""

import os
import pickle
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(texts: List[str], chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=len,
		separators=["\n\n", "\n", ".", " "]
	)
	chunks = []
	for t in texts:
		chunks.extend(splitter.split_text(t))
	return chunks


def build_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
	model = SentenceTransformer(model_name)
	embeddings = model.encode(chunks, show_progress_bar=True)
	return np.array(embeddings).astype("float32")


def build_faiss_index(embedding_matrix: np.ndarray) -> faiss.IndexFlatL2:
	dim = embedding_matrix.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(embedding_matrix)
	return index


def save_vector_store(index: faiss.IndexFlatL2, metadata: List[Dict[str, Any]], chunks: List[str], out_dir: str = "vectorstore") -> None:
	"""Save the FAISS index and metadata. For compatibility, write to both
	`vectorstore/` (rubric) and `vector_store/` (existing notebooks).
	"""
	# primary dir per rubric
	os.makedirs(out_dir, exist_ok=True)
	faiss.write_index(index, os.path.join(out_dir, "complaint_index.faiss"))
	with open(os.path.join(out_dir, "metadata.pkl"), "wb") as f:
		pickle.dump(metadata, f)
	with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
		pickle.dump(chunks, f)

	# also write to legacy path for compatibility
	legacy = "vector_store"
	os.makedirs(legacy, exist_ok=True)
	faiss.write_index(index, os.path.join(legacy, "complaint_index.faiss"))
	with open(os.path.join(legacy, "metadata.pkl"), "wb") as f:
		pickle.dump(metadata, f)
	with open(os.path.join(legacy, "chunks.pkl"), "wb") as f:
		pickle.dump(chunks, f)


if __name__ == "__main__":
	# convenience CLI: look for filtered CSV then build index
	import pandas as pd

	filtered_path = os.path.join("data", "filtered_complaints_sampled.csv")
	if not os.path.exists(filtered_path):
		filtered_path = os.path.join("data", "filtered_complaints.csv")
	if not os.path.exists(filtered_path):
		raise FileNotFoundError("Place filtered CSV at data/filtered_complaints.csv or filtered_complaints_sampled.csv first.")

	df = pd.read_csv(filtered_path)
	if "Cleaned_Narrative" not in df.columns:
		raise ValueError("Filtered CSV must contain column 'Cleaned_Narrative'")

	all_chunks = []
	metadata = []
	for idx, row in df.reset_index().iterrows():
		chunks = chunk_documents([row["Cleaned_Narrative"]])
		all_chunks.extend(chunks)
		metadata.extend([{"product": row.get("Product"), "complaint_id": row.get("Complaint ID"), "original_index": int(row.get("index", idx))}] * len(chunks))

	embeddings = build_embeddings(all_chunks)
	index = build_faiss_index(embeddings)
	save_vector_store(index, metadata, all_chunks, out_dir="vectorstore")
	print(f"Saved FAISS index and metadata to vectorstore/ and vector_store/")

