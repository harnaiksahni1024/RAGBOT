from langchain_groq import ChatGroq
import streamlit as st
import os
from ragas import evaluate
from ragas.metrics import (
    faithfulness
)
from datasets import Dataset

def evaluate_ragas_results():
    history = st.session_state.chat_history

    if not history :
        st.warning("No Chat History to Evaluate")
        return

    question,answer = None,None
    for speaker,msg in reversed(history):
        if speaker.lower() == 'human' and not question:
            question = msg.strip()
        elif speaker.lower()== 'ai' and not answer:
            answer = msg.split('\n\n')[0].strip() # as we only need answer part not confidence
        if question and answer:
            break

    if not question and answer:
        st.warning("Could not Extract question and answer for evaluation")
        return
            

    docs = st.session_state.get('last_sources', [])
    if not docs:
        st.warning("No Documents retrieved for this question")

    context = [doc.page_content for doc in docs[:3] if doc.page_content.strip()]            
    if not context:
        st.warning("Retrieved Documents are empty")
        return
    
  

    data = Dataset.from_dict({
        'question':[question],
        'answer':[answer],
        'contexts':[context]
    })

    groq_llm =ChatGroq(
        model='llama3-8b-8192',
        groq_api_key =os.environ.get("GROQ_API_KEY") 
    )
    with st.spinner("Evaluating answer quality using RAGAS"):
        try:
            results=evaluate(
                data,
                metrics = [faithfulness],
                llm=groq_llm
            )
            st.success("Evaluation Complete")
        except Exception as e:
             st.error(f"Evaluation Failed: {str(e)}")
             return
        
    df = results.to_pandas()
    st.write("### Evaluation Results DataFrame:")
    st.dataframe(df)

    for col in df.columns:
        if col != 'question':
            st.metric(f"{col.replace('_', ' ').capitalize()} Score", round(df[col][0], 3))
  

    # Reuse the original context for display (not from df)
    st.markdown("## Top Retrieved Contexts:")
    for i, c in enumerate(context):
        with st.expander(f"Context {i+1}"):
            st.write(c.strip() if c.strip() else "Empty Context")