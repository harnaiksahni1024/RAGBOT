from langchain.document_loaders import TextLoader,PyPDFLoader,UnstructuredFileLoader,Docx2txtLoader
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




#creating a function to save these files in a temporary location
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("temp_files",uploaded_file.name)
    os.makedirs("temp_files",exist_ok=True)
    with open(file_path,'wb') as f:
        f.write(uploaded_file.read())
    return file_path

    