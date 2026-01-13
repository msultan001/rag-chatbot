import streamlit as st
import sys
import os

# Add src to path so we can import RAGChatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_chatbot import RAGChatbot

# Page configuration
st.set_page_config(
    page_title="CrediTrust - Financial Analyst Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize RAG Chatbot
@st.cache_resource
def load_chatbot():
    return RAGChatbot()

chatbot = load_chatbot()

# App styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .clear-button>button {
        background-color: #f44336;
    }
    .source-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196F3;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.title("🤖 CrediTrust RAG Assistant")
st.subheader("Interactive Complaint Analysis")

st.info("Greetings! I am your AI assistant specialized in analyzing financial complaints. Ask me anything about customer feedback or product issues.")

# Layout: Two columns for input/actions
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input("Enter your question here:", placeholder="e.g., What are the main issues with credit cards?", key="query_input")

with col2:
    st.write("### Actions")
    submit_btn = st.button("Submit")
    clear_btn = st.button("Clear Chat", key="clear", help="Clear the current view")

# Handle Clear Button
if clear_btn:
    st.session_state.query_input = ""
    st.rerun()

# Handle Submission
if submit_btn and user_input:
    with st.spinner("Analyzing complaints and generating an answer..."):
        response = chatbot.ask(user_input)
        
        # Display Answer
        st.markdown(f"### 💬 Answer")
        st.write(response["answer"])

        # Display Sources
        if response["sources"]:
            with st.expander("📚 View Source Snippets"):
                for i, src in enumerate(response["sources"]):
                    st.markdown(f"**Source {i+1}** (Product: {src['metadata'].get('product', 'N/A')})")
                    st.markdown(f"<div class='source-box'>{src['content']}</div>", unsafe_allow_html=True)
        else:
            st.warning("No source snippets found for this query.")
elif submit_btn and not user_input:
    st.warning("Please enter a question first.")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This RAG pipeline uses SentenceTransformers for embeddings, FAISS for retrieval, and DistilGPT2 for generation.")
st.sidebar.markdown("© 2026 CrediTrust Intelligence")
