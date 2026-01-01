import streamlit as st
import requests

API_URL = "http://api:8000/ask"

st.set_page_config(page_title="RAG Assistant", page_icon="🧠")

st.title("🧠 RAG Assistant")
st.caption("Ask questions based on your documents")

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.text_input("Ask a question:")

if st.button("Ask") and question.strip():
    st.session_state.chat.append(("You", question))

    try:
        res = requests.post(API_URL, json={"question": question}, timeout=120)
        res.raise_for_status()
        answer = res.json()["answer"]
    except Exception as e:
        answer = f"❌ Error: {e}"

    st.session_state.chat.append(("Assistant", answer))

st.divider()

for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"🧠 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Assistant:** {msg}")
