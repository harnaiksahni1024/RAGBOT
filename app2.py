#use of streamlit to display 
import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from modules.process_documents import process_documents,count_tokens
from modules.embedding import get_text_chunks
from modules.vector_store import get_vector_store,load_embeddings
from modules.chains import get_conversational_chain
from modules.handle_input import handle_input
from modules.evaluation import evaluate_ragas_results




import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
#application heading


def main():
    load_dotenv()
    st.set_page_config(page_title="🧠 RAGBOT - Chat with Documents", layout='centered')
    
    # === Top Banner ===
    
    st.markdown("### 📄 Ask Questions from Your Documents Using RAG + LLM")
    st.caption("Upload PDF, DOCX, or TXT files. The chatbot will retrieve and answer based on content using LLaMA-3.")

    # === Session Init ===
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # === Instructions ===
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        1. Upload one or more documents using the sidebar.
        2. Set chunk size and overlap if needed.
        3. Click **Submit and Process**.
        4. Ask questions in the chat input.
        5. Click **Evaluate Answer Quality (RAGAS)** to assess the response.
        """)

    # === Chat Input ===
    st.divider()
    st.header("💬 Chat with your uploaded documents")

    user_question = st.chat_input("Ask your question here...")
    if user_question:
        if st.session_state.conversation:
            handle_input(user_question)
        else:
            st.warning("⚠️ Please upload and process documents first!")

    # === Sidebar Upload and Settings ===
    with st.sidebar:
        st.title("📂 Document Settings")
        upload_docs = st.file_uploader("📤 Upload files", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)
        if upload_docs:
            st.success(f"{len(upload_docs)} file(s) uploaded!")

        with st.expander("⚙️ Advanced Chunking Settings"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
            chunk_overlap = st.slider("Chunk Overlap", 100, 400, 200, step=50)

        if st.button("✅ Submit and Process") and upload_docs:
            with st.spinner("🔄 Processing documents..."):
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                if os.path.exists("temp_files"):
                    shutil.rmtree("temp_files")
                chunks = process_documents(upload_docs, chunk_size, chunk_overlap)
                embeddings = load_embeddings()
                vector_store = get_vector_store(chunks, embeddings)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("✅ Documents processed successfully!")
                total_tokens = sum(count_tokens(chunk.page_content) for chunk in chunks)
                st.toast(f"💡 {len(chunks)} chunks | {total_tokens} total tokens")

    # === Conversation History ===
    if st.session_state.chat_history:
        with st.expander("📜 View Conversation History"):
            for speaker, msg in st.session_state.chat_history:
                st.markdown(f"**{speaker}:** {msg}")

    # === Evaluate ===
    st.divider()
    if st.button("🔍 Evaluate Answer Quality (RAGAS)"):
        evaluate_ragas_results()


if __name__ =="__main__":
    main()