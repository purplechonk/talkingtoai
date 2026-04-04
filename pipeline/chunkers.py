# pipeline/chunkers.py
# All chunking strategies in one place.
# Each function takes a list of corpus documents and returns a flat list of chunks
# in the same format your existing clean_chunks.json uses, so the rest of the
# pipeline (embedder, Qdrant upsert) requires zero changes.
 
import uuid
import hashlib
import tiktoken
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
 
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import config
 
enc = tiktoken.get_encoding("cl100k_base")
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _count_tokens(text: str) -> int:
    return len(enc.encode(text))
 
 
def _stable_id(doc: dict, strategy: str, text: str) -> str:
    """
    Deterministic chunk ID derived from doc content_hash + strategy + text.
    Returned as a UUID string (required by Qdrant — must be uint or UUID).
    Same source text + same strategy always produces the same ID, so
    re-running notebook 02 only upserts changed/new chunks — no duplicates.
    """
    key = f"{doc.get('content_hash', doc.get('url', ''))}-{strategy}-{text[:200]}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return str(uuid.UUID(hex=digest))


def _make_chunk(text: str, doc: dict, strategy: str, parent_id: Optional[str] = None) -> dict:
    """Build a chunk dict that matches the existing clean_chunks.json schema."""
    return {
        "chunk_id":      _stable_id(doc, strategy, text),
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
 
 
# ── Strategy 3: Sentence-window chunking ─────────────────────────────────────

def chunk_sentence_window(corpus: list, window_size: int = config.SENTENCE_WINDOW_SIZE) -> list:
    """
    Each chunk is a single sentence stored as the retrieval unit, plus a
    surrounding window of sentences stored as 'window_text' for the LLM.

    Retrieval (embedding)  → precise: just the one sentence
    LLM context            → rich:    window_size sentences before + sentence + window_size after

    Typically outperforms parent_child on exact policy-rule Q&A because
    embedding a single sentence is more focused than embedding a 400-token block.
    No API calls required — zero extra cost.
    """
    import re

    # Sentence boundary: split after . ! ? followed by whitespace
    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    chunks = []
    for doc in corpus:
        raw_sentences = _SENT_RE.split(doc["full_text"])
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        for i, sentence in enumerate(sentences):
            if len(sentence) < config.MIN_CHUNK_CHARS:
                continue

            start = max(0, i - window_size)
            end   = min(len(sentences), i + window_size + 1)
            window_text = " ".join(sentences[start:end])

            chunk = _make_chunk(sentence, doc, strategy="sentence_window")
            chunk["window_text"] = window_text
            chunk["window_size"] = window_size
            chunks.append(chunk)

    return chunks


# ── Strategy 4: Proposition chunking ─────────────────────────────────────────

_PROPOSITION_SYSTEM = """\
You are a precise text analyser. Given a passage, extract ALL atomic propositions from it.

An atomic proposition is:
  - A single, self-contained declarative statement
  - Contains exactly one fact, rule, number, or claim
  - Understandable without reading the surrounding text
  - Written as a complete sentence in the same language as the passage

Return ONLY valid JSON in this exact format (no markdown):
{"propositions": ["proposition 1", "proposition 2", "..."]}"""


def chunk_proposition(
    corpus: list,
    openai_api_key: str,
    base_chunk_size: int = 512,
) -> list:
    """
    LLM-based atomic proposition extraction.

    Pipeline:
      1. Split each document into passages using recursive splitting
      2. For each passage, call the LLM to extract atomic propositions
      3. Each proposition becomes its own chunk, with 'source_passage' stored
         as context for the LLM at query time

    Trade-off: highest retrieval precision of all strategies, but costs
    roughly 1 LLM call per ~512-token passage. Run after recursive to
    validate the pipeline first.

    Requires openai_api_key.
    """
    import json
    import time
    from openai import OpenAI

    oai = OpenAI(api_key=openai_api_key)

    passage_splitter = RecursiveCharacterTextSplitter(
        chunk_size=base_chunk_size,
        chunk_overlap=50,
        length_function=_count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in corpus:
        passages = passage_splitter.split_text(doc["full_text"])

        for passage in passages:
            if len(passage) < config.MIN_CHUNK_CHARS:
                continue

            try:
                response = oai.chat.completions.create(
                    model=config.LLM_MODEL,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": _PROPOSITION_SYSTEM},
                        {"role": "user",   "content": f"Extract propositions:\n\n{passage}"},
                    ],
                    response_format={"type": "json_object"},
                )
                raw  = response.choices[0].message.content.strip()
                data = json.loads(raw)

                # Accept {"propositions": [...]} or any single-key dict wrapping a list
                if isinstance(data, dict):
                    propositions = data.get("propositions") or next(
                        (v for v in data.values() if isinstance(v, list)), []
                    )
                else:
                    propositions = data

                for prop in propositions:
                    prop = prop.strip() if isinstance(prop, str) else ""
                    if len(prop) < 20:
                        continue
                    chunk = _make_chunk(prop, doc, strategy="proposition")
                    chunk["source_passage"] = passage   # richer context for the LLM
                    chunks.append(chunk)

                time.sleep(0.2)   # stay well within rate limits

            except Exception as exc:
                print(f"  [warn] proposition extraction failed — falling back to passage: {exc}")
                chunks.append(_make_chunk(passage, doc, strategy="proposition"))
 
 
# ── Strategy 5: Parent-child chunking ────────────────────────────────────────
 
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
        chunks = get_chunks("recursive",        corpus)
        chunks = get_chunks("semantic",         corpus, openai_api_key=OPENAI_API_KEY)
        chunks = get_chunks("sentence_window",  corpus)
        chunks = get_chunks("parent_child",     corpus)
        chunks = get_chunks("proposition",      corpus, openai_api_key=OPENAI_API_KEY)
    """
    needs_key = ("semantic", "proposition")
    if strategy in needs_key and not openai_api_key:
        raise ValueError(f"'{strategy}' chunking requires openai_api_key")

    if strategy == "recursive":
        return chunk_recursive(corpus)
    elif strategy == "semantic":
        return chunk_semantic(corpus, openai_api_key)
    elif strategy == "sentence_window":
        return chunk_sentence_window(corpus)
    elif strategy == "parent_child":
        return chunk_parent_child(corpus)
    elif strategy == "proposition":
        return chunk_proposition(corpus, openai_api_key)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {config.CHUNKING_STRATEGIES}")