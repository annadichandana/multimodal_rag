from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

emb_old = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
doc = [Document(page_content="test")]
vector_store = FAISS.from_documents(doc, emb_old)
vector_store.save_local("test_mismatch_index")

emb_new = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vs = FAISS.load_local("test_mismatch_index", emb_new, allow_dangerous_deserialization=True)

try:
    vs.similarity_search("test")
except Exception as e:
    print(f"Exception printed: '{e}'")
