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


def save_vector_store(chunks: List[str], metadata: List[Dict[str, Any]], out_dir: str = "vectorstore") -> None:
    """
    Saves the chunks and metadata using LangChain's FAISS wrapper for easy loading in the chatbot.
    """
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embeddings, metadatas=metadata)
    
    # Save to primary dir
    os.makedirs(out_dir, exist_ok=True)
    vector_db.save_local(out_dir)
    print(f"Saved FAISS index to {out_dir}/")

    # Also save to legacy path for notebook compatibility
    legacy = "vector_store"
    os.makedirs(legacy, exist_ok=True)
    vector_db.save_local(legacy)
    print(f"Saved FAISS index to {legacy}/")

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
    
    # Simple chunking for demo purposes
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for idx, row in df.reset_index().iterrows():
        text = str(row["Cleaned_Narrative"])
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
        metadata.extend([{"product": row.get("Product"), "complaint_id": row.get("Complaint ID")}] * len(chunks))

    save_vector_store(all_chunks, metadata, out_dir="vectorstore")
    print(f"Successfully built and saved vector store artifacts.")

