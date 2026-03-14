import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

class SimpleMemory:
    def __init__(self):
        self.buffer = ""
    def add_user_message(self, msg):
        self.buffer += f"Human: {msg}\n"
    def add_ai_message(self, msg):
        self.buffer += f"AI: {msg}\n"

from src.text_indexer import load_text_index, load_bm25_index
from src.image_indexer import load_image_index
from src.table_indexer import load_table_index
from main import answer_query

load_dotenv()

st.set_page_config(page_title="Multimodal RAG Agent", layout="wide")
st.title("🤖 Multimodal RAG with Hybrid Search & Reranking")

if "memory" not in st.session_state:
    st.session_state.memory = SimpleMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_llms():
    text_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Missing GOOGLE_API_KEY in .env file.")
        st.stop()
    return ChatGoogleGenerativeAI(model=text_model, google_api_key=google_api_key)

@st.cache_resource
def load_indexes():
    try:
        ti = load_text_index("text_faiss_index")
        bi, bd = load_bm25_index("text_faiss_index")
    except Exception:
        ti, bi, bd = None, None, None
        
    try:
        ii = load_image_index("image_faiss_index")
    except Exception:
        ii = None
        
    try:
        tabi=load_table_index("table_faiss_index")
    except Exception:
        tabi = None
        
    return ti, ii, tabi, bi, bd

llm = get_llms()
text_idx, image_idx, table_idx, bm25_idx, bm25_docs = load_indexes()

if not any([text_idx, image_idx, table_idx]):
    st.warning("No FAISS indexes found. Please run `python main.py --file <path_to_pdf>` in your terminal first to index a document.")
    st.stop()

# Display Chat History from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about your indexed document..."):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context using Hybrid Search & Reranking..."):
            try:
                answer = answer_query(
                    query=prompt,
                    llm=llm,
                    text_index=text_idx,
                    image_index=image_idx,
                    table_index=table_idx,
                    bm25_index=bm25_idx,
                    bm25_docs=bm25_docs,
                    memory=st.session_state.memory
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err_msg = str(e).strip() or repr(e)
                st.error(f"Error answering query: {err_msg}")
