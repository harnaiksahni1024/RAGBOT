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
    load_dotenv()
    st.set_page_config("RAGBOT",layout='centered')
    st.title("Q&A RAGBOT")
    st.caption("Upload PDF,TXT or DOCX files.The Chatbot will Retrieve and give answer based on content")
    
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation= None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_sources' not in st.session_state:
        st.session_state.last_sources = []

    user_question = st.chat_input("Ask your question here ...")
    if user_question:
        if st.session_state.conversation:
            handle_input(user_question)
        else:
            st.warning("First upload the document in the Sidebar")



    with st.sidebar:
        st.title("Options and Upload")
        retrieval_type = st.selectbox('Retrieval Strategy',['mmr','similarity'],index=0)
        temperature  = st.slider("LLM Temperature",0.0,1.0,0.2)
        upload_docs = st.file_uploader("Upload your file and click on submit",type=['pdf','txt','docx'],accept_multiple_files=True)
        if upload_docs:
            st.success("Documents Uploaded")
            for file in upload_docs:
                st.markdown(f"{file.name}")
        

        if st.button("Submit and Process") and upload_docs:
            with st.spinner("Processing ..."):
                if os.path.exists("Faiss_Index"):
                    shutil.rmtree('Faiss_Index')
                
                
                st.session_state.chat_history = []
                st.session_state.last_sources = []
                st.session_state.conversation = None
                
                chunks = process_documents(upload_docs,chunk_size=1000,chunk_overlap=200)
                embeddings = load_embeddings()
                vector_store = get_vector_store(chunks,embeddings)

                
                st.session_state.conversation = get_conversational_chain(
                    vector_store=vector_store,
                    temperature=temperature,
                    search_type=retrieval_type
                    )
                st.session_state.last_sources = chunks
                st.success("Documents processed successfully!")
                total_tokens = sum(count_tokens(chunk.page_content) for chunk in chunks)
                st.info(f"Total Chunks: {len(chunks)} | Total Tokens: {total_tokens}")
    
      # to view Conversation History
    if st.session_state.chat_history:
        with st.expander(" View Conversation History"):
            for speaker, msg in st.session_state.chat_history:
                st.markdown(f"**{speaker}:** {msg}")
    
    
    
    if st.button("Evaluate your response"):        
        evaluate_ragas_results()




           
if __name__ =="__main__":
    main()