



from langchain_groq import ChatGroq
import streamlit as st
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from datasets import Dataset




def evaluate_ragas_results():
    history = st.session_state.chat_history
    if not history:
        st.warning("No chat history to evaluate.")
        return

    # Extract the last Q&A pair from history
    question, answer = "", ""
    for speaker, msg in reversed(history):
        if speaker == "Human" and not question:
            question = msg
        elif speaker == "AI" and not answer:
            answer = msg.split("\n\n")[0]  # Exclude confidence part
        if question and answer:
            break

    if not question or not answer:
        st.warning("Could not extract Q&A for evaluation.")
        return

    # Retrieve top 3 relevant documents
    retriever = st.session_state.conversation.retriever
    docs = retriever.get_relevant_documents(question)
    context = [doc.page_content for doc in docs[:3] if doc.page_content.strip()]

    # Prepare dataset for RAGAS
    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [context],
    })

    # Setup Groq LLM
    groq_llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )


    # Evaluate with RAGAS
    with st.spinner("Evaluating answer quality using RAGAS..."):
        results = evaluate(
            data,
            metrics=[faithfulness],
            llm=groq_llm  # Use Groq-wrapped LLM
        )
        st.success("Evaluation complete!")

        # Display results
    df = results.to_pandas()
    st.dataframe(df)


    for metric in df.columns:
        st.metric(label=metric.capitalize(), value=round(df[metric][0], 3))



