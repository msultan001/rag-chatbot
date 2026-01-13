"""Core RAG Chatbot logic using LangChain and FAISS.

This module provides the RAGChatbot class which:
- Loads a FAISS vector store
- Initializes a lightweight LLM (distilgpt2)
- Sets up a RetrievalQA chain for answering user queries based on context.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    """
    RAG Chatbot implementation using LangChain and FAISS.
    """

    def __init__(
        self, 
        vector_store_path: str = "vectorstore", 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "distilgpt2"
    ):
        """
        Initializes the RAG Chatbot.

        Args:
            vector_store_path: Path to the FAISS index directory.
            model_name: Embedding model name.
            llm_model: Lightweight LLM model for generation.
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.llm_model = llm_model
        self.rag_chain = None
        self.vector_db = None

        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """
        Sets up the embedding model, vector store, and RAG chain.
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}...")
            embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

            if not os.path.exists(self.vector_store_path):
                # Check for legacy path
                legacy = "vector_store"
                if os.path.exists(legacy):
                    self.vector_store_path = legacy
                    logger.info(f"Using legacy vector store path: {legacy}")
                else:
                    logger.error(f"Vector store directory not found: {self.vector_store_path}")
                    return

            logger.info(f"Loading vector store from {self.vector_store_path}...")
            self.vector_db = FAISS.load_local(
                self.vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )

            logger.info(f"Initializing LLM: {self.llm_model}...")
            # Use GPU if available
            device = 0 if torch.cuda.is_available() else -1
            llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                device=device,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                truncation=True
            )
            llm = HuggingFacePipeline(pipeline=llm_pipeline)

            rag_prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are a financial analyst assistant at CrediTrust.
Your task is to answer customer complaint-related questions using only the provided context.

Context:
{context}

Question:
{question}

Answer (based only on the context above):
"""
            )

            logger.info("Setting up RetrievalQA chain...")
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
                chain_type="stuff",
                chain_type_kwargs={"prompt": rag_prompt_template},
                return_source_documents=True
            )
            logger.info("RAG pipeline setup successful.")

        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Asks a question to the chatbot.

        Args:
            query: The user's question.

        Returns:
            Dict[str, Any]: Contains 'answer' and 'sources'.
        """
        if not self.rag_chain:
            return {
                "answer": "Error: RAG pipeline is not initialized. Please ensure the vector store exists.",
                "sources": []
            }

        try:
            logger.info(f"Processing query: {query}")
            result = self.rag_chain.invoke({"query": query})
            
            answer = result.get("result", "No answer generated.")
            source_docs = result.get("source_documents", [])
            
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in source_docs
            ]

            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return {
                "answer": f"Error: {e}",
                "sources": []
            }

if __name__ == "__main__":
    # Quick CLI test
    bot = RAGChatbot()
    test_query = "What common complaints exist about credit cards?"
    response = bot.ask(test_query)
    print(f"\nQ: {test_query}\nA: {response['answer']}\n")
    print(f"Sources retrieved: {len(response['sources'])}")
