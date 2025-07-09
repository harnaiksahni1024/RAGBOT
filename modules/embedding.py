from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

@st.cache_resource(show_spinner = "Load HuggingFace Embeddings ....")
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L12-v2")
    return embeddings

@st.cache_resource(show_spinner = "Creating Faiss Vector Store")
def get_vector_store(_text_chunks,_embeddings):
    vs = FAISS.from_documents(_text_chunks,_embeddings)
    vs.save_local("Faiss_Index")
    return vs

