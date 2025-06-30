from dotenv import load_dotenv
import pdfplumber
import docx2txt
import shutil
import os
import tiktoken
from datetime import datetime

from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# ========= LOADERS WITH METADATA ==========
def load_txt_loader(file_path, file_name):
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    for doc in docs:
        doc.metadata['source_file'] = file_name
    return docs

def load_pdf_loader(file_path, file_name):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['source_file'] = file_name
    return docs

def load_docx_loader(file_path, file_name):
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['source_file'] = file_name
    return docs

def load_unstructured(file_path, file_name):
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['source_file'] = file_name
    return docs

def save_uploaded_file(uploaded_file):
    file_path = os.path.join("temp_files", uploaded_file.name)
    os.makedirs("temp_files", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

# ========== TEXT SPLITTING ==========
def get_text_chunks(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# ========== EMBEDDINGS & VECTOR ==========
@st.cache_resource(show_spinner="Loading HuggingFace Embeddings...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

@st.cache_resource(show_spinner="Creating FAISS Vector Store...")
def get_vector_store(_text_chunks, _embeddings):
    vs = FAISS.from_documents(_text_chunks, _embeddings)
    vs.save_local("faiss_index")
    return vs

# ========== PARSER & OUTPUT STRUCTURE ==========
class QaResponse(BaseModel):
    answer: str = Field(..., description="The complete answer to the user's question")
    confidence: str = Field(..., description="Confidence level of the answer (High/Medium/Low)")

parser = PydanticOutputParser(pydantic_object=QaResponse)

# ========== LLM CHAIN ==========
def get_conversational_chain(vector_store):
    system_msg = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant. Use ONLY the context below to answer the user's question. "
        "If the context is not sufficient, say 'I don't have enough context.'\n\n{format_instructions}"
    )

    human_msg = HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion:\n{question}"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg]).partial(
        format_instructions=parser.get_format_instructions())

    llm = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name='llama3-8b-8192')

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        retriever=vector_store.as_retriever(search_type='mmr'),
        return_source_documents=True
    )

# ========== HANDLE USER INPUT ==========
def handle_input(user_question):
    chain = st.session_state.conversation
    result = chain({'question': user_question})
    answer = result['answer']
    sources = result.get('source_documents', [])

    try:
        parsed = parser.parse(answer)
    except Exception:
        st.warning("Answer format not as expected. Showing raw output.")
        parsed = QaResponse(answer=answer, confidence='Unknown')

    st.session_state.chat_history.append(("Human", user_question))
    st.session_state.chat_history.append(("AI", parsed.answer + f"\n\nConfidence: {parsed.confidence}"))

    for speaker, msg in st.session_state.chat_history:
        with st.chat_message("AI" if speaker == 'AI' else 'HUMAN'):
            st.markdown(msg)

    # Show source file
    file_names = set(doc.metadata.get('source_file', 'Unknown') for doc in sources)
    if file_names:
        st.info("Answer derived from:\n" + "\n".join(f"- {f}" for f in file_names))

# ========== PROCESS FILES ==========
@st.cache_data(show_spinner='Processing Documents...')
def process_documents(files, chunk_size=1000, chunk_overlap=200):
    documents = []
    for file in files:
        file_path = save_uploaded_file(file)
        try:
            if file.name.endswith(".pdf"):
                documents.extend(load_pdf_loader(file_path, file.name))
            elif file.name.endswith(".docx"):
                documents.extend(load_docx_loader(file_path, file.name))
            elif file.name.endswith(".txt"):
                documents.extend(load_txt_loader(file_path, file.name))
            else:
                documents.extend(load_unstructured(file_path, file.name))
        except:
            st.warning(f"Standard loader failed. Using unstructured loader for {file.name}")
            documents.extend(load_unstructured(file_path, file.name))

    return get_text_chunks(documents, chunk_size, chunk_overlap)

# ========== TOKEN COUNT ==========
def count_tokens(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

# ========== MAIN APP ==========
def main():
    load_dotenv()
    st.set_page_config(page_title="RAGBOT", layout='centered')
    st.title("Ask your question")
    st.header("Chat with different files using RAG")

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = "All"

    # Handle chat input
    user_question = st.chat_input("Ask the question from uploaded file")
    if user_question:
        if st.session_state.conversation:
            handle_input(user_question)
        else:
            st.warning("First upload the document in sidebar.")

    # Sidebar UI
    with st.sidebar:
        st.title("Menu:")
        upload_docs = st.file_uploader("Upload your file(s) and click on submit", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

        if upload_docs:
            st.write("### Files Uploaded:")
            for file in upload_docs:
                st.markdown(f"- {file.name}")

            # File filter dropdown
            uploaded_filenames = [file.name for file in upload_docs]
            st.session_state.selected_file = st.selectbox("Filter by file (optional):", ["All"] + uploaded_filenames)

        with st.expander("Advanced Setting"):
            chunk_size = st.slider("Chunk_Size", 500, 2000, 1000)
            chunk_overlap = st.slider("Chunk_Overlap", 100, 400, 200)

        # Submit button
        if st.button("Submit and Process") and upload_docs:
            with st.spinner("Processing..."):
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                if os.path.exists("temp_files"):
                    shutil.rmtree("temp_files")

                chunks = process_documents(upload_docs, chunk_size, chunk_overlap)
                st.session_state.all_chunks = chunks  # Store all chunks

                # Filter by selected file
                if st.session_state.selected_file != "All":
                    chunks = [c for c in chunks if c.metadata.get("source_file") == st.session_state.selected_file]

                embeddings = load_embeddings()
                vector_store = get_vector_store(chunks, embeddings)
                st.session_state.conversation = get_conversational_chain(vector_store)

                st.success("Documents processed successfully!")
                total_tokens = sum(count_tokens(chunk.page_content) for chunk in chunks)
                st.info(f"Total Chunks: {len(chunks)} | Total Tokens: {total_tokens}")

if __name__ == "__main__":
    main()
