"""
text_indexer.py
---------------
Builds and queries a FAISS vector index for plain-text chunks.

Same embedding approach as Project 1 — this is the text modality index.

We reuse the all-MiniLM-L6-v2 sentence-transformer model because:
  * It is fast and runs fully locally (no API calls, no cost).
  * Its 384-dimensional embeddings strike a good balance between quality
    and memory / speed.
  * It has proven strong retrieval performance on diverse Q&A benchmarks.

The only difference from Project 1 is that this index is *one of three*
indexes in the multimodal pipeline.  The query router decides whether to
hit this index, the image index, the table index, or all three.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle
import os


# Shared embedding model — instantiated once to avoid repeated model loading.
_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance for all-MiniLM-L6-v2."""
    return HuggingFaceEmbeddings(model_name=_EMBED_MODEL_NAME)


def index_text_chunks(
    text_blocks: list[str],
    index_path: str = "text_faiss_index",
) -> FAISS:
    """
    Embed a list of text strings and persist them as a FAISS index.

    Parameters
    ----------
    text_blocks : Raw text strings (one per page, paragraph, or chunk).
    index_path  : Directory path where the FAISS index files are saved.

    Returns
    -------
    A LangChain FAISS vector store ready for similarity search.
    """
    if not text_blocks:
        raise ValueError("text_blocks is empty — nothing to index.")

    # Wrap each string in a LangChain Document so we can store metadata.
    # We record the chunk number so retrieved results can be traced back.
    docs = [
        Document(page_content=block, metadata={"chunk_id": i, "modality": "text"})
        for i, block in enumerate(text_blocks)
    ]

    embeddings = _get_embeddings()

    # FAISS.from_documents embeds all docs in a single batch and builds
    # the index in memory, then we persist it to disk.
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(index_path)

    # Build and save BM25 index
    tokenized_corpus = [doc.page_content.lower().split(" ") for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_path = f"{index_path}_bm25"
    os.makedirs(bm25_path, exist_ok=True)
    with open(os.path.join(bm25_path, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(bm25_path, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

    print(f"[text_indexer] Indexed {len(docs)} text chunks FAISS → '{index_path}' & BM25 → '{bm25_path}'")
    return vector_store, bm25, docs


def load_text_index(index_path: str) -> FAISS:
    """
    Load a previously saved FAISS text index from disk.

    Parameters
    ----------
    index_path : Directory path that was passed to index_text_chunks().

    Returns
    -------
    A LangChain FAISS vector store.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[text_indexer] Loaded text index from '{index_path}'")
    return vector_store

def load_bm25_index(index_path: str):
    """Load BM25 objects and docs from disk."""
    bm25_path = f"{index_path}_bm25"
    try:
        with open(os.path.join(bm25_path, "bm25.pkl"), "rb") as f:
            bm25 = pickle.load(f)
        with open(os.path.join(bm25_path, "docs.pkl"), "rb") as f:
            docs = pickle.load(f)
        print(f"[text_indexer] Loaded BM25 index from '{bm25_path}'")
        return bm25, docs
    except Exception as e:
        print(f"[text_indexer] Could not load BM25 index from '{bm25_path}': {e}")
        return None, None


def search_text(query: str, vector_store: FAISS, bm25_index=None, bm25_docs=None, k: int = 3) -> list:
    """
    Retrieve the top-k most relevant text chunks using Hybrid Search (FAISS + BM25).

    Parameters
    ----------
    query        : Natural language question or search string.
    vector_store : A loaded or freshly-built FAISS text index.
    bm25_index   : BM25Okapi instance for keyword search.
    bm25_docs    : List of Documents used by BM25.
    k            : Number of results to return.

    Returns
    -------
    List of (Document, score) tuples ordered by descending similarity.
    (Score transformed to behave like FAISS L2 distance so lower = better).
    """
    faiss_results = vector_store.similarity_search_with_score(query, k=k)
    
    if bm25_index is None or bm25_docs is None:
        return faiss_results
        
    tokenized_query = query.lower().split(" ")
    bm25_top_n = bm25_index.get_top_n(tokenized_query, bm25_docs, n=k)
    
    # Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    
    for rank, (doc, _) in enumerate(faiss_results):
        doc_id = doc.metadata.get("chunk_id")
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
        
    for rank, doc in enumerate(bm25_top_n):
        doc_id = doc.metadata.get("chunk_id")
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
        
    id_to_doc = {doc.metadata.get("chunk_id"): doc for doc in bm25_docs}
    sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Inverse RRF score so lower is "better" for downstream compatibility
    hybrid_results = [(id_to_doc[doc_id], 1.0 / rrf_scores[doc_id]) for doc_id in sorted_docs[:k]]
    return hybrid_results
