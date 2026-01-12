import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class RAGChatbot:
    def __init__(self, vector_store_path: str = "vectorstore", model_name: str = "distilgpt2"):
        """
        Initializes the RAG chatbot by loading the vector store and setting up the LLM chain.
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load vector store
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {vector_store_path}. Please run src/chunking_embedding.py first.")
        
        print(f"Loading vector store from {vector_store_path}...")
        self.vector_db = FAISS.load_local(
            vector_store_path, 
            self.embeddings,
            allow_dangerous_deserialization=True # Required for loading pickle-based FAISS local
        )
        
        # Setup LLM
        print(f"Initializing LLM ({model_name})...")
        hf_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=50256 # For distilgpt2
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        # Setup Prompt Template
        template = """
You are a financial analyst assistant at CrediTrust.
Your task is to answer customer complaint-related questions using only the provided context.
If the answer is not in the context, say that you don't know based on the provided information.

Context:
{context}

Question:
{question}

Answer (based only on the context above):
"""
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Setup RetrievalQA chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Queries the RAG pipeline.
        """
        response = self.rag_chain({"query": query})
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }

if __name__ == "__main__":
    # Quick test if run as script
    try:
        bot = RAGChatbot()
        test_query = "What are the common issues with credit reporting?"
        result = bot.ask(test_query)
        print(f"\nQuery: {test_query}")
        print(f"Answer: {result['answer']}")
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"- {doc.page_content[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
