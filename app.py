import streamlit as st
from dotenv import load_dotenv
import os
import shutil


from modules.handle_input import handle_input
from modules.process_documents import process_documents,count_tokens
from modules.embedding import get_vector_store,load_embeddings
from modules.chain import get_conversational_chain
from modules.evaluation import evaluate_ragas_results



import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
#application heading




def main():
    st.set_page_config("RAGBOT")
    load_dotenv()
    st.title("Q&A RAGBOT")
    if 'conversation' not in st.session_state:
        st.session_state.conversation= None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []



    st.header("Chat with your uploaded file")



    user_question = st.chat_input("Ask the Question")
    if user_question:
        if st.session_state.conversation:
            handle_input(user_question)
        else:
            st.warning("First upload the document in the Sidebar")



    with st.sidebar:
        st.title("Menu :")
        upload_docs = st.file_uploader("Upload your file and click on submit",type=['pdf','txt','docx'],accept_multiple_files=True)
        if upload_docs:
            st.write("Documents Uploaded:")
            for file in upload_docs:
                st.markdown(f"{file.name}")



        with st.expander("Advance Setting for Chunks"):
            chunk_size = st.slider("Chunk Size",500,2000,1000)
            chunk_overlap = st.slider("Chunk Overlap",100,400,200)

        


        if st.button("submit and process") and upload_docs:
            with st.spinner("Processing ..."):
                if os.path.exists("faiss_index"):
                    shutil.rmtree('faiss_index')
                if os.path.exists("temp_files"):
                    shutil.rmtree('temp_files')
                
                chunks = process_documents(upload_docs,chunk_size,chunk_overlap)
                embeddings = load_embeddings()
                vector_store = get_vector_store(chunks,embeddings)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Documents processed successfully!")
                total_tokens = sum(count_tokens(chunk.page_content) for chunk in chunks)
                st.info(f"Total Chunks: {len(chunks)} | Total Tokens: {total_tokens}")
    if st.button("Evaluate your response"):        
        evaluate_ragas_results()




           
if __name__ =="__main__":
    main()