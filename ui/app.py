import streamlit as st
import os
import sys

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_chatbot import RAGChatbot

# Page config
st.set_page_config(page_title="Financial Analyst RAG Chatbot", layout="wide")

st.title("💸 CrediTrust: Financial Analyst Assistant")
st.markdown("""
Welcome to the CrediTrust Complaint Assistant. 
Ask questions about customer complaints and I will look them up in our database to provide accurate answers.
""")

# Initialize the chatbot only once
@st.cache_resource
def load_chatbot():
    try:
        return RAGChatbot(vector_store_path="vectorstore")
    except Exception as e:
        st.error(f"Failed to load chatbot: {e}. Check if the vector store exists.")
        return None

chatbot = load_chatbot()

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        if chat["role"] == "assistant" and "sources" in chat:
            with st.expander("Source Chunks"):
                for doc in chat["sources"]:
                    st.write(f"- {doc.page_content}")
                    st.write(f"  *Product: {doc.metadata.get('product')}*")
                    st.divider()

# Input area
if query := st.chat_input("Ask a question about financial complaints..."):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if chatbot:
        with st.spinner("Analyzing complaints..."):
            try:
                result = chatbot.ask(query)
                answer = result["answer"]
                sources = result["source_documents"]
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                # Render response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Source Chunks"):
                        for doc in sources:
                            st.write(f"- {doc.page_content}")
                            st.write(f"  *Product: {doc.metadata.get('product')}*")
                            st.divider()
            except Exception as e:
                st.error(f"Error during query: {e}")
    else:
        st.warning("Chatbot is not loaded. Please ensure the vector store is built.")

# Footer
st.divider()
st.caption("Powered by RAG Pipeline | LangChain | FAISS | HuggingFace")
