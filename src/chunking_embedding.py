"""Chunking, embedding and FAISS vector store helpers.

Provides programmatic equivalents of the notebook steps:
- chunk cleaned narratives with LangChain's RecursiveCharacterTextSplitter
- create embeddings with SentenceTransformers `all-MiniLM-L6-v2`
- build and persist a FAISS index plus metadata
"""

import os
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def chunk_documents(texts: List[str], chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    """
    Chunks a list of texts into smaller segments using a recursive character splitter.

    Args:
        texts: List of strings to chunk.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = []
        for t in texts:
            if isinstance(t, str):
                chunks.extend(splitter.split_text(t))
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        return []

def build_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generates embeddings for a list of text chunks using a SentenceTransformer model.

    Args:
        chunks: List of text chunks.
        model_name: Name of the SentenceTransformer model to use.

    Returns:
        np.ndarray: Embedding matrix.
    """
    if not chunks:
        logger.warning("No chunks provided for embedding generation.")
        return np.array([])
        
    try:
        logger.info(f"Loading embedding model: {model_name}...")
        model = SentenceTransformer(model_name)
        logger.info("Generating embeddings...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        return np.array(embeddings).astype("float32")
    except Exception as e:
        logger.error(f"Failed to build embeddings: {e}")
        return np.array([])

def build_faiss_index(embedding_matrix: np.ndarray) -> Optional[faiss.IndexFlatL2]:
    """
    Creates a FAISS index from an embedding matrix.

    Args:
        embedding_matrix: Matrix of embeddings.

    Returns:
        Optional[faiss.IndexFlatL2]: The FAISS index, or None if creation fails.
    """
    if embedding_matrix.size == 0:
        logger.error("Empty embedding matrix provided for FAISS index build.")
        return None
        
    try:
        dim = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embedding_matrix)
        logger.info(f"Built FAISS index with dimension {dim}")
        return index
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        return None

def save_vector_store(chunks: List[str], metadata: List[Dict[str, Any]], out_dir: str = "vectorstore") -> None:
    """
    Saves the chunks and metadata using LangChain's FAISS wrapper for easy loading in the chatbot.

    Args:
        chunks: List of text chunks.
        metadata: List of metadata dictionaries corresponding to each chunk.
        out_dir: Directory where the vector store will be saved.
    """
    if not chunks:
        logger.error("No chunks to save to vector store.")
        return

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        
        logger.info(f"Initializing HuggingFaceEmbeddings for FAISS save in {out_dir}...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_db = FAISS.from_texts(chunks, embeddings, metadatas=metadata)
        
        # Save to primary dir
        os.makedirs(out_dir, exist_ok=True)
        vector_db.save_local(out_dir)
        logger.info(f"Saved FAISS index to {out_dir}/")

        # Also save to legacy path for notebook compatibility (optional, but keeping as per original)
        legacy = "vector_store"
        os.makedirs(legacy, exist_ok=True)
        vector_db.save_local(legacy)
        logger.info(f"Saved FAISS index to {legacy}/")
        
    except ImportError:
        logger.error("Required LangChain packages not found. Please install langchain-community and langchain-huggingface.")
    except Exception as e:
        logger.error(f"Failed to save vector store: {e}")

def run_indexing_pipeline(input_path: str, output_dir: str = "vectorstore") -> None:
    """
    Runs the full indexing pipeline: load data -> chunk -> embed -> index -> save.
    """
    if not os.path.exists(input_path):
        logger.error(f"Data file not found at {input_path}")
        return

    try:
        logger.info(f"Reading data from {input_path}...")
        df = pd.read_csv(input_path)
        if "Cleaned_Narrative" not in df.columns:
            logger.error("Filtered CSV must contain column 'Cleaned_Narrative'")
            return

        all_chunks = []
        metadata = []
        
        logger.info("Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for _, row in df.reset_index().iterrows():
            text = str(row["Cleaned_Narrative"])
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            # Carry over product and ID for context
            meta = {
                "product": row.get("Product", "Unknown"),
                "complaint_id": row.get("Complaint ID", "Unknown")
            }
            metadata.extend([meta] * len(chunks))

        logger.info(f"Generated {len(all_chunks)} chunks. Saving to vector store...")
        save_vector_store(all_chunks, metadata, out_dir=output_dir)
        logger.info("Successfully built and saved vector store artifacts.")
        
    except Exception as e:
        logger.error(f"Error in indexing pipeline: {e}")

if __name__ == "__main__":
    # Look for filtered CSV then build index
    target_csv = os.path.join("data", "filtered_complaints_sampled.csv")
    if not os.path.exists(target_csv):
        target_csv = os.path.join("data", "filtered_complaints.csv")
        
    run_indexing_pipeline(target_csv)

