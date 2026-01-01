import streamlit as st
import requests

# ===============================
# CONFIG
# ===============================
API_URL = "http://localhost:8000"  # Backend must be running

st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# UI HEADER
# ===============================
st.title("📚 RAG Assistant")
st.markdown("Upload a document and ask questions based on its content.")

# ===============================
# FILE UPLOAD
# ===============================
st.subheader("📄 Upload a TXT document")

uploaded_file = st.file_uploader(
    "Choose a .txt file",
    type=["txt"]
)

if uploaded_file is not None:
    with st.spinner("Uploading and indexing document..."):
        try:
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue())
            }

            response = requests.post(
                f"{API_URL}/upload",
                files=files,
                timeout=60
            )

            if response.status_code == 200:
                st.success(response.json().get("message", "Uploaded successfully"))
            else:
                st.error(response.text)

        except Exception as e:
            st.error(f"Upload failed: {e}")

# ===============================
# QUESTION INPUT
# ===============================
st.divider()
st.subheader("💬 Ask a question")

question = st.text_input("Enter your question")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=120
            )

            if response.status_code == 200:
                answer = response.json().get("answer", "")
                st.session_state.chat_history.append((question, answer))
            else:
                st.error(response.text)

        except Exception as e:
            st.error(f"Request failed: {e}")

# ===============================
# CHAT HISTORY
# ===============================
st.divider()
st.subheader("🧠 Chat History")

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
    st.markdown("---")
