from modules.embedding import get_vector_store
from modules.output_parser import parser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_groq import ChatGroq
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st


def get_conversational_chain(vector_store):
    system_msg = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant. "
    "Use ONLY the context below to answer the user's question. "
    "Do NOT hallucinate or make up any information. "
    "If the context is not sufficient, say 'I don't have enough context.' "
    "Respond in the following format:\n\n"
    "{format_instructions}"
    )

    human_msg = HumanMessagePromptTemplate.from_template(
    "Context :\n{context}\n"
    "Question : \n{question}\n"      
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_msg,human_msg]).partial(format_instructions = parser.get_format_instructions())
    
    llm  = ChatGroq(groq_api_key = os.environ.get("GROQ_API_KEY"),model_name = 'llama3-8b-8192')

    memory= ConversationBufferMemory(memory_key = "chat_history",return_messages=True,output_key='answer')

    chain =ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        combine_docs_chain_kwargs={'prompt':chat_prompt},
        retriever = vector_store.as_retriever(search_type ='mmr',search_kwargs={"k": 3, "score_threshold": 0.6}),
        return_source_documents=True
    )

    return chain
