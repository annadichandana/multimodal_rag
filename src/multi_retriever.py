"""
multi_retriever.py
------------------
Queries one or more FAISS indexes in parallel based on the query types
returned by the router, then merges the results into a single ranked list.

The challenge of ranking across modalities
-------------------------------------------
Each FAISS index returns an L2 distance score in the embedding space of
all-MiniLM-L6-v2 (384 dimensions).  Because all three indexes use the *same*
embedding model, scores are theoretically comparable — but in practice:

  * The distribution of scores differs by modality (short captions tend to
    have lower variance than long text chunks).
  * A "0.3 score" for a text chunk may not be semantically equivalent to a
    "0.3 score" for an image caption.

Two ranking strategies are discussed here:

  Simple (implemented): interleave results — 1 text result, 1 image result,
  1 table result — so every modality is represented in the context, regardless
  of raw score magnitude.  Easy to implement, transparent to the user.

  Complex (alternative): normalise scores per-modality using min-max scaling,
  then sort globally.  More precise but can still suppress a modality entirely
  if its scores are consistently higher (worse) than others.

We use the simple interleaving approach and let the generator model weight
results contextually via its attention mechanism.

De-duplication
--------------
The same text snippet can theoretically appear in multiple indexes (e.g. a
table that was also mentioned verbatim in the text).  We de-duplicate on
content string to avoid feeding the same information twice to the generator.
"""

from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from .query_router import QueryType
from .text_indexer import search_text
from .image_indexer import search_images
from .table_indexer import search_tables

_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANKER_MODEL_NAME)
    return _reranker


def retrieve_all(
    query: str,
    query_types: list[QueryType],
    text_index: FAISS | None,
    image_index: FAISS | None,
    table_index: FAISS | None,
    bm25_index = None,
    bm25_docs = None,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve top-k results from each relevant index and return a combined list.
    """
    results: list[dict] = []

    if QueryType.TEXT in query_types and text_index is not None:
        for doc, score in search_text(query, text_index, bm25_index, bm25_docs, k=k):
            results.append(
                {
                    "content": doc.page_content,
                    "modality": "text",
                    "metadata": doc.metadata,
                    "source": f"text_chunk_{doc.metadata.get('chunk_id', '?')}",
                    "score": float(score),
                }
            )

    if QueryType.IMAGE in query_types and image_index is not None:
        for item in search_images(query, image_index, k=k):
            results.append(
                {
                    "content": item["caption"],
                    "modality": "image",
                    "metadata": {
                        "image_path": item["image_path"],
                        "image_type": item["image_type"],
                    },
                    "source": item["image_path"],
                    "score": item["score"],
                }
            )

    if QueryType.TABLE in query_types and table_index is not None:
        for item in search_tables(query, table_index, k=k):
            results.append(
                {
                    "content": item["description"],
                    "modality": "table",
                    "metadata": {
                        "table_id": item["table_id"],
                        "csv_path": item["csv_path"],
                        "page": item["page"],
                    },
                    "source": item["table_id"],
                    "score": item["score"],
                }
            )

    return results


def merge_and_rank_results(query: str, results: list[dict], top_n: int = 5) -> list[dict]:
    """
    De-duplicate and route results through the BAAI CrossEncoder Reranker.
    """
    # De-duplicate on content string.
    seen_content: set[str] = set()
    unique: list[dict] = []
    for r in results:
        if r["content"] not in seen_content:
            seen_content.add(r["content"])
            unique.append(r)

    if not unique:
        return []
        
    print(f"  [multi_retriever] Reranking {len(unique)} candidate chunks...")
    reranker = _get_reranker()
    pairs = [[query, r["content"]] for r in unique]
    scores = reranker.predict(pairs)
    
    for r, score in zip(unique, scores):
        r["rerank_score"] = float(score)
        
    unique.sort(key=lambda x: x["rerank_score"], reverse=True)
    return unique[:top_n]
