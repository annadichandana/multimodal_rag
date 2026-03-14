from app import get_llms, load_indexes, SimpleMemory
from main import answer_query

llm = get_llms()
text_idx, image_idx, table_idx, bm25_idx, bm25_docs = load_indexes()
memory = SimpleMemory()

try:
    print("Query 1...")
    answer1 = answer_query(
        query="What is the document about?",
        llm=llm,
        text_index=text_idx,
        image_index=image_idx,
        table_index=table_idx,
        bm25_index=bm25_idx,
        bm25_docs=bm25_docs,
        memory=memory
    )
    print("Query 2...")
    answer2 = answer_query(
        query="What was the revenue in Q4 and how does it compare with Q1?",
        llm=llm,
        text_index=text_idx,
        image_index=image_idx,
        table_index=table_idx,
        bm25_index=bm25_idx,
        bm25_docs=bm25_docs,
        memory=memory
    )
    print("SUCCESS!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error answering query: {e}")
