#now we have to handle the user input
from modules.chain import get_conversational_chain
import streamlit as st
from modules.output_parser import parser,QaResponse


def handle_input(user_question):
    chain = st.session_state.conversation           #intialize the chain
    result = chain.invoke({'question':user_question})
    answer = result['answer']
    sources = result.get('source_documents', [])

    try:
        parsed = parser.parse(answer)
    except Exception:
        st.warning("Answer format not as expected. Showing raw output.")
        parsed = QaResponse(answer=answer, confidence='Unknown')

    #intialize chat history
    st.session_state.chat_history.append(("HUMAN", user_question))
    st.session_state.chat_history.append(("AI", parsed.answer + f"\n\nConfidence: {parsed.confidence}"))

    for speaker,msg in st.session_state.chat_history:
        with st.chat_message("AI" if speaker == "AI" else "HUMAN"):
            st.markdown(msg)
    
    contexts = [doc.page_content for doc in sources if doc.page_content.strip() != '']
    if not contexts:
        st.session_state.chat_history.append(("Human", user_question))
        st.session_state.chat_history.append(("AI", "I don’t have enough context to answer that question."))
        st.warning("No relevant context found for the question.")
        return
