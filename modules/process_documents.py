from modules.file_handler import (load_docx_loader,load_txt_loader,load_pdf_loader,load_unstructured,save_uploaded_file)
from modules.chunks import get_text_chunks
import streamlit as st
import tiktoken

#process the documents 
@st.cache_data(show_spinner='Processing Documents...')
def process_documents(files,chunk_size=1000,chunk_overlap=200):
    documents=[]
    for file in files:
        file_path = save_uploaded_file(file)
        try:
            if file.name.endswith(".pdf"):
                documents.extend(load_pdf_loader(file_path))
            elif file.name.endswith(".txt"):
                documents.extend(load_txt_loader(file_path))
            elif file.name.endswith(".docx"):
                documents.extend(load_docx_loader(file_path))
            else :
                raise ValueError("Unsupported file type") 
        except:
            st.warning(f"standard file loader failed, using unstructured file loader for {file.name}")
            documents.extend(load_unstructured(file_path))
    return get_text_chunks(documents,chunk_size,chunk_overlap)            


#done for calculating no of chunks and tokens
def count_tokens(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    return len(tokenizer.encode(text))