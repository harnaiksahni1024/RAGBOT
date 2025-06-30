from langchain_text_splitters import RecursiveCharacterTextSplitter

#here we convert uploaded documents into chunks which helps in reading the file in better way


def get_text_chunks(text,chunk_size=1000,chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(text)

