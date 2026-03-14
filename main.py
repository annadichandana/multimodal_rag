"""
main.py — Multimodal RAG Pipeline
-----------------------------------
Orchestrates the full multimodal retrieval-augmented generation pipeline:
  1. Parse PDF → extract text, images, tables
  2. Index each modality in its own FAISS vector store
  3. Route a user query to the relevant index(es)
  4. Retrieve top-k results
  5. Generate a grounded answer

Usage examples
--------------
# Full pipeline (index + query)
python main.py --file data/sample_docs/annual_report.pdf --query "What was Q4 revenue?"

# Skip image captioning to save GPT-4V cost during development
python main.py --file data/sample_docs/annual_report.pdf --query "Summarise the findings" --skip-images

# Interactive mode — ask multiple questions after indexing once
python main.py --file data/sample_docs/annual_report.pdf --interactive
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

class SimpleMemory:
    def __init__(self):
        self.buffer = ""
    def add_user_message(self, msg):
        self.buffer += f"Human: {msg}\n"
    def add_ai_message(self, msg):
        self.buffer += f"AI: {msg}\n"

from src.multimodal_parser import parse_document
from src.text_indexer import index_text_chunks
from src.image_processor import process_all_images
from src.image_indexer import index_image_captions
from src.table_processor import process_all_tables
from src.table_indexer import index_table_descriptions
from src.query_router import classify_query
from src.multi_retriever import retrieve_all, merge_and_rank_results
from src.generator import generate_answer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal RAG: answer questions over text, images, and tables in a PDF."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the PDF document to process.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Question to answer.  Required unless --interactive is set.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model for text generation (default: env GEMINI_MODEL or gemini-2.5-flash).",
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        dest="vision_model",
        help="Gemini vision model for image captioning (default: env VISION_MODEL or gemini-2.5-flash).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        dest="skip_images",
        help="Skip image captioning (saves cost/time during development).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        dest="skip_tables",
        help="Skip LLM-based table description generation.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After indexing, enter an interactive Q&A loop.",
    )
    return parser


def answer_query(
    query: str,
    llm,
    text_index,
    image_index,
    table_index,
    bm25_index=None,
    bm25_docs=None,
    memory=None
) -> str:
    """Route → retrieve → rerank → generate for a single query."""
    print(f"\n[main] Query: {query}")

    query_types = classify_query(query, llm)
    print(f"[main] Router selected modalities: {[qt.value for qt in query_types]}")

    raw_results = retrieve_all(
        query=query,
        query_types=query_types,
        text_index=text_index,
        image_index=image_index,
        table_index=table_index,
        bm25_index=bm25_index,
        bm25_docs=bm25_docs,
        k=10,  # pull more chunks to allow reranker to do its job
    )

    ranked_results = merge_and_rank_results(query, raw_results, top_n=5)
    print(f"[main] Retained top {len(ranked_results)} result(s) after BM25/FAISS retrieval and reranking.")

    chat_history = memory.buffer if memory else ""
    answer = generate_answer(query, ranked_results, llm, chat_history=chat_history)
    
    if memory:
        memory.add_user_message(query)
        memory.add_ai_message(answer)
        
    return answer


def main() -> None:
    load_dotenv()

    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.interactive and args.query is None:
        parser.error("--query is required unless --interactive is set.")

    if not os.path.isfile(args.file):
        print(f"[main] ERROR: File not found: {args.file}")
        sys.exit(1)

    text_model = args.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    vision_model = args.vision_model or os.getenv("VISION_MODEL", "gemini-2.5-flash")
    images_dir = os.getenv("IMAGES_OUTPUT_DIR", "data/extracted/images")
    tables_dir = os.getenv("TABLES_OUTPUT_DIR", "data/extracted/tables")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("[main] ERROR: GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(model=text_model, google_api_key=google_api_key)
    vision_llm = ChatGoogleGenerativeAI(model=vision_model, google_api_key=google_api_key)
    print(f"\n[main] Parsing document: {args.file}")
    doc = parse_document(args.file, images_dir=images_dir, tables_dir=tables_dir)

    print(
        f"[main] Found {len(doc.text_blocks)} text blocks, "
        f"{len(doc.image_paths)} images, "
        f"{len(doc.tables)} tables."
    )
    text_index = None
    bm25_index, bm25_docs = None, None
    if doc.text_blocks:
        print(f"\n[main] Indexing {len(doc.text_blocks)} text blocks …")
        text_index, bm25_index, bm25_docs = index_text_chunks(doc.text_blocks, index_path="text_faiss_index")
    else:
        print("[main] No text blocks found — skipping text index.")

    image_index = None
    if not args.skip_images and doc.image_paths:
        print(f"\n[main] Captioning {len(doc.image_paths)} image(s) with {vision_model} …")
        print("       Use --skip-images during development to avoid LLM charges.")
        image_data = process_all_images(doc.image_paths, vision_llm)
        print(f"\n[main] Indexing {len(image_data)} image caption(s) …")
        image_index = index_image_captions(image_data, index_path="image_faiss_index")
    elif args.skip_images:
        print("\n[main] --skip-images set: skipping image captioning and indexing.")
    else:
        print("\n[main] No images found in document.")

    table_index = None
    if not args.skip_tables and doc.tables:
        print(f"\n[main] Processing {len(doc.tables)} table(s) …")
        table_data = process_all_tables(doc.tables, llm, tables_dir=tables_dir)
        print(f"[main] Indexing {len(table_data)} table description(s) …")
        table_index = index_table_descriptions(table_data, index_path="table_faiss_index")
    elif args.skip_tables:
        print("\n[main] --skip-tables set: skipping table processing and indexing.")
    else:
        print("\n[main] No tables found in document.")

    print("\n" + "─" * 60)

    if args.interactive:
        print("[main] Interactive mode. Type 'quit' or 'exit' to stop.\n")
        memory = SimpleMemory()
        while True:
            try:
                query = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[main] Exiting.")
                break
            if query.lower() in ("quit", "exit", "q"):
                print("[main] Exiting.")
                break
            if not query:
                continue
            answer = answer_query(query, llm, text_index, image_index, table_index, bm25_index, bm25_docs, memory)
            print(f"\nAnswer:\n{answer}\n")
            print("─" * 60)
    else:
        answer = answer_query(args.query, llm, text_index, image_index, table_index, bm25_index, bm25_docs)
        print(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    main()
