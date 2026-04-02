# pipeline/chunkers.py
# All chunking strategies in one place.
# Each function takes a list of corpus documents and returns a flat list of chunks
# in the same format your existing clean_chunks.json uses, so the rest of the
# pipeline (embedder, Qdrant upsert) requires zero changes.
 
import uuid
import tiktoken
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
 
import sys
sys.path.append("..")
import config
 
enc = tiktoken.get_encoding("cl100k_base")
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _count_tokens(text: str) -> int:
    return len(enc.encode(text))
 
 
def _make_chunk(text: str, doc: dict, strategy: str, parent_id: Optional[str] = None) -> dict:
    """Build a chunk dict that matches the existing clean_chunks.json schema."""
    return {
        "chunk_id":      str(uuid.uuid4()),
        "text":          text,
        "token_count":   _count_tokens(text),
        "source":        doc.get("source"),
        "category":      doc.get("category"),
        "region":        doc.get("region"),
        "type":          doc.get("type"),
        "title":         doc.get("title"),
        "url":           doc.get("url"),
        "scraped_at":    doc.get("scraped_at"),
        "strategy":      strategy,
        # parent_id is only set for the parent_child strategy
        "parent_id":     parent_id,
    }
 
 
def _filter_short(chunks: list, min_chars: int = config.MIN_CHUNK_CHARS) -> list:
    return [c for c in chunks if len(c["text"]) >= min_chars]
 
 
# ── Strategy 1: Recursive (replacement for your existing custom chunker) ──────
 
def chunk_recursive(corpus: list) -> list:
    """
    LangChain RecursiveCharacterTextSplitter.
    Tries \\n\\n → \\n → '. ' → ' ' in order — same idea as your hand-built
    chunker but more battle-tested. token-based via tiktoken.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=_count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in corpus:
        texts = splitter.split_text(doc["full_text"])
        for t in texts:
            chunks.append(_make_chunk(t, doc, strategy="recursive"))
    return _filter_short(chunks)
 
 
# ── Strategy 2: Semantic chunking ─────────────────────────────────────────────
 
def chunk_semantic(corpus: list, openai_api_key: str) -> list:
    """
    LangChain SemanticChunker.
    Embeds each sentence with text-embedding-3-small, then splits wherever
    cosine similarity drops sharply between adjacent sentences (topic shift).
    Produces semantically coherent chunks at the cost of variable chunk size.
 
    Note: this calls the OpenAI embedding API for every sentence in the corpus
    — it's slower and costs more than recursive. Cache the result.
    """
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=openai_api_key,
    )
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",  # split at the biggest similarity drops
        breakpoint_threshold_amount=95,           # top 5% of drops become split points
    )
    chunks = []
    for doc in corpus:
        texts = splitter.split_text(doc["full_text"])
        for t in texts:
            chunks.append(_make_chunk(t, doc, strategy="semantic"))
    return _filter_short(chunks)
 
 
# ── Strategy 3: Header-based (structure-aware) ────────────────────────────────
 
def chunk_header(corpus: list) -> list:
    """
    Splits on Markdown-style section headings (# / ## / ###).
    Works best for your MOM/WSH policy docs which have clear section structure.
    Falls back to recursive splitting for documents without headings.
 
    Your scraper already produces plain text, so this strategy first tries to
    detect heading patterns (lines that are short, title-cased, and followed
    by a blank line) and inserts # markers before splitting.
    """
    import re
 
    def _inject_headings(text: str) -> str:
        """Heuristically promote short all-caps or title-case lines to headings."""
        lines = text.split("\n")
        out   = []
        for line in lines:
            stripped = line.strip()
            # Treat short (< 80 char), non-sentence lines as headings
            if (
                stripped
                and len(stripped) < 80
                and not stripped.endswith(".")
                and (stripped.isupper() or stripped.istitle())
            ):
                out.append(f"## {stripped}")
            else:
                out.append(line)
        return "\n".join(out)
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=_count_tokens,
        separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
    )
 
    chunks = []
    for doc in corpus:
        marked_text = _inject_headings(doc["full_text"])
        texts = splitter.split_text(marked_text)
        for t in texts:
            # Strip the injected ## markers from the stored text
            clean = re.sub(r"^##?\s+", "", t, flags=re.MULTILINE).strip()
            chunks.append(_make_chunk(clean, doc, strategy="header"))
    return _filter_short(chunks)
 
 
# ── Strategy 4: Parent-child chunking ────────────────────────────────────────
 
def chunk_parent_child(corpus: list) -> list:
    """
    Stores two levels of chunks:
    - Small child chunks (~CHUNK_SIZE tokens) for precise retrieval
    - Large parent chunks (~PARENT_CHUNK_SIZE tokens) stored in payload so
      the LLM gets more context when generating the answer
 
    During retrieval you search over child chunks (small = precise), but
    pass the parent text to the LLM (large = more context).
    The parent_id field links each child to its parent.
 
    Returns both parent and child chunks in the list.
    Parent chunks have strategy="parent", children have strategy="parent_child".
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=_count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=_count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
 
    all_chunks = []
    for doc in corpus:
        parent_texts = parent_splitter.split_text(doc["full_text"])
        for parent_text in parent_texts:
            parent_id    = str(uuid.uuid4())
            parent_chunk = _make_chunk(parent_text, doc, strategy="parent")
            parent_chunk["chunk_id"] = parent_id
 
            child_texts = child_splitter.split_text(parent_text)
            children    = []
            for child_text in child_texts:
                if len(child_text) < config.MIN_CHUNK_CHARS:
                    continue
                child = _make_chunk(child_text, doc, strategy="parent_child")
                child["parent_id"]   = parent_id
                child["parent_text"] = parent_text   # stored for LLM context
                children.append(child)
 
            if children:
                all_chunks.append(parent_chunk)
                all_chunks.extend(children)
 
    return all_chunks
 
 
# ── Dispatcher ────────────────────────────────────────────────────────────────
 
def get_chunks(strategy: str, corpus: list, openai_api_key: str = None) -> list:
    """
    Main entry point. Call this from notebooks.
 
    Usage:
        from pipeline.chunkers import get_chunks
        chunks = get_chunks("recursive", corpus)
        chunks = get_chunks("semantic",  corpus, openai_api_key=OPENAI_API_KEY)
    """
    if strategy == "recursive":
        return chunk_recursive(corpus)
    elif strategy == "semantic":
        if not openai_api_key:
            raise ValueError("semantic chunking requires openai_api_key")
        return chunk_semantic(corpus, openai_api_key)
    elif strategy == "header":
        return chunk_header(corpus)
    elif strategy == "parent_child":
        return chunk_parent_child(corpus)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {config.CHUNKING_STRATEGIES}")