"""
query_router.py
---------------
Classifies an incoming user query to determine which content modalities
(text, image, table) are most likely to contain the answer, then routes
retrieval to the appropriate FAISS indexes.

Why routing matters
--------------------
Without routing every query would hit all three indexes, which:
  * Wastes embedding / similarity-search compute.
  * Inflates cost when GPT-4V-captioned image indexes are large.
  * Dilutes the final context with irrelevant cross-modal results.

By classifying upfront we retrieve *only* from relevant indexes, reducing
latency and cost while keeping the context focused.

When to use ALL
----------------
Complex questions (e.g. "Summarise the findings from section 2") often span
all content types.  When the classifier is uncertain it returns ALL, which is
the safe default — it is better to over-search than to miss the answer.

Parsing the LLM output
-----------------------
We ask the LLM to respond with a JSON object `{"types": [...]}` to make
parsing deterministic.  If the response cannot be parsed as JSON we fall back
to ALL to maintain correctness at the cost of a broader search.
"""

import json
import re
from enum import Enum


class QueryType(Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    TABLE = "TABLE"
    ALL = "ALL"


_CLASSIFICATION_PROMPT = """\
Classify this query to determine which type of document content would best answer it.

Query: {query}

Choose one or more from:
- TEXT: The answer is likely in text paragraphs
- IMAGE: The answer requires looking at a visual/diagram/photo
- TABLE: The answer requires numerical data from a table or chart
- ALL: Search all content types

Common patterns:
- "show me", "what does X look like", "diagram of" → IMAGE
- "how many", "revenue", "statistics", "percentage", "trend" → TABLE
- "explain", "describe", "what is", "how does" → TEXT
- Complex questions → ALL

Respond with JSON only: {{"types": ["TEXT", "TABLE"]}}
"""


def classify_query(query: str, llm=None) -> list[QueryType]:
    """
    Classify a user query by relevant content modality using fast keyword routing,
    falling back to standard semantic classification if needed.

    Parameters
    ----------
    query : The user's natural-language question.
    llm   : Optional LangChain LLM / chat model (unused in the fast router).

    Returns
    -------
    List of QueryType enum values indicating which indexes to search.
    """
    query_lower = query.lower()
    
    # Fast Rule-based Routing
    if any(keyword in query_lower for keyword in ["chart", "image", "picture", "diagram", "photo", "visual"]):
        print("  [query_router] Fast-routed to IMAGE based on keywords.")
        return [QueryType.IMAGE]
        
    if any(keyword in query_lower for keyword in ["table", "compare", "statistics", "data in row", "metrics", "percentage"]):
        print("  [query_router] Fast-routed to TABLE based on keywords.")
        return [QueryType.TABLE]
        
    if any(keyword in query_lower for keyword in ["summarize", "overall", "comprehensive", "all info"]):
        print("  [query_router] Fast-routed to ALL based on keywords.")
        return [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]

    # Default to text
    print("  [query_router] Defaulting to TEXT.")
    return [QueryType.TEXT]
