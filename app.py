import streamlit as st
from rag import get_qa_chain

st.title("💊 Pharma RAG Assistant (Groq + Free Embeddings)")

qa = get_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask your question")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    result = qa(user_input)
    answer = result["result"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(answer)
