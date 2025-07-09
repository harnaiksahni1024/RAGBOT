from langchain_community.document_loaders import TextLoader,PyPDFLoader,UnstructuredFileLoader,Docx2txtLoader
import os

#how to handle files
    
def load_txt_loader(file_path):
    loader = TextLoader(file_path,encoding='utf-8')
    documents= loader.load()
    return documents

#pdf files
def load_pdf_loader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


#docx files

def load_docx_loader(file_path):
    loader = Docx2txtLoader(file_path)
    documents=loader.load()
    return documents


#unstructured loader
def load_unstructured(file_path):
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    return documents



import tempfile

def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1]  # Keep the correct file extension
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    return temp_file.name
    