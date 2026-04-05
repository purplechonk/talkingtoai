# pipeline/retriever.py
# All retrieval strategies in one place.
# Your existing dense + hybrid logic is preserved exactly.
# New LangChain strategies (multi_query, compression) are added alongside.
 
import torch
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import config


def _multi_query_retriever_cls():
    """LangChain 1.x moved retrievers out of `langchain.retrievers`; use langchain-classic fallback."""
    try:
        from langchain.retrievers import MultiQueryRetriever

        return MultiQueryRetriever
    except ImportError:
        try:
            from langchain_classic.retrievers.multi_query import MultiQueryRetriever

            return MultiQueryRetriever
        except ImportError as e:
            raise ImportError(
                'multi_query needs retriever classes. Install: pip install "langchain-classic>=1.0"'
            ) from e


def _compression_retriever_components():
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor

        return ContextualCompressionRetriever, LLMChainExtractor
    except ImportError:
        try:
            from langchain_classic.retrievers.contextual_compression import (
                ContextualCompressionRetriever,
            )
            from langchain_classic.retrievers.document_compressors.chain_extract import (
                LLMChainExtractor,
            )

            return ContextualCompressionRetriever, LLMChainExtractor
        except ImportError as e:
            raise ImportError(
                'compression needs retriever classes. Install: pip install "langchain-classic>=1.0"'
            ) from e


# ── Low-level search helpers (your existing code, unchanged) ──────────────────
 
def get_dense_vec(query: str, oai: OpenAI) -> list:
    return oai.embeddings.create(model=config.EMBEDDING_MODEL, input=[query]).data[0].embedding
 
 
def get_sparse_vec(query: str, bm25_model: SparseTextEmbedding):
    return list(bm25_model.embed([query]))[0]
 
 
def _build_filter(category=None, source=None, region=None,
                  role_relevance=None, answer_type=None):
    filter_map = {
        "category":       category,
        "source":         source,
        "region":         region,
        "role_relevance": role_relevance,
        "answer_type":    answer_type,
    }
    conditions = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in filter_map.items() if v is not None
    ]
    return models.Filter(must=conditions) if conditions else None
 
 
def _format_results(results: list, search_mode: str, query: str,
                    top_k: int, filters_applied: dict) -> dict:
    retrieved_chunks = []
    for rank, r in enumerate(results, 1):
        p     = r.payload
        score = getattr(r, "score", None)
        retrieved_chunks.append({
            "chunk_id":       p.get("chunk_id"),
            "rank":           rank,
            "score":          round(score, 4) if isinstance(score, float) else None,
            "title":          p.get("title"),
            "source":         p.get("source"),
            "category":       p.get("category"),
            "answer_type":    p.get("answer_type"),
            "role_relevance": p.get("role_relevance"),
            "topic_tags":     p.get("topic_tags", []),
            "url":            p.get("url"),
            "text":           p.get("text"),
            # Context priority:
            #   parent_child  → parent_text  (large parent passage)
            #   sentence_window → window_text (surrounding sentences)
            #   proposition   → source_passage (original passage the proposition came from)
            #   all others    → text itself
            "context_text":   (
                p.get("parent_text")
                or p.get("window_text")
                or p.get("source_passage")
                or p.get("text")
            ),
        })
    return {
        "query":            query,
        "filters_applied":  filters_applied,
        "search_mode":      search_mode,
        "top_k":            top_k,
        "retrieved_chunks": retrieved_chunks,
    }
 
 
# ── Strategy 1: Dense (your existing code) ────────────────────────────────────
 
def dense_search(
    query: str, oai: OpenAI, qdrant: QdrantClient,
    collection_name: str, filt=None, top_k: int = config.TOP_K,
) -> list:
    vec = get_dense_vec(query, oai)
    return qdrant.query_points(
        collection_name=collection_name,
        query=vec,
        using="dense",
        query_filter=filt,
        limit=top_k,
        with_payload=True,
    ).points
 
 
# ── Strategy 2: Hybrid RRF (your existing code) ───────────────────────────────
 
def hybrid_search(
    query: str, oai: OpenAI, qdrant: QdrantClient, bm25_model: SparseTextEmbedding,
    collection_name: str, filt=None, top_k: int = config.TOP_K,
) -> list:
    vec    = get_dense_vec(query, oai)
    sp_emb = get_sparse_vec(query, bm25_model)
    return qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=vec, using="dense", limit=top_k * 4, filter=filt),
            models.Prefetch(
                query=SparseVector(
                    indices=sp_emb.indices.tolist(),
                    values=sp_emb.values.tolist(),
                ),
                using="sparse",
                limit=top_k * 4,
                filter=filt,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points
 
 
# ── Strategy 3: Multi-query (LangChain) ───────────────────────────────────────
 
def multi_query_search(
    query: str, oai: OpenAI, qdrant: QdrantClient,
    collection_name: str, openai_api_key: str,
    filt=None, top_k: int = config.TOP_K,
) -> list:
    """
    Rewrites the query 3 ways using an LLM, runs each variant through dense
    search, then merges and deduplicates results.
    Helps when queries are ambiguous or phrased in an unusual way.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore

    MultiQueryRetriever = _multi_query_retriever_cls()
    lc_embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=openai_api_key)
    vectorstore   = QdrantVectorStore(
        client=qdrant,
        collection_name=collection_name,
        embedding=lc_embeddings,
        vector_name="dense",
    )
    llm      = ChatOpenAI(model=config.LLM_MODEL, api_key=openai_api_key, temperature=0)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        llm=llm,
    )
    docs = retriever.invoke(query)
 
    # Convert LangChain Document objects back to Qdrant-style point dicts
    # so the rest of the pipeline (_format_results) works unchanged
    seen   = set()
    points = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
 
            class _FakePoint:
                payload = {**doc.metadata, "text": doc.page_content}
                score   = None
 
            points.append(_FakePoint())
 
    return points[:top_k]
 
 
# ── Strategy 4: Contextual compression (LangChain) ───────────────────────────
 
def compression_search(
    query: str, oai: OpenAI, qdrant: QdrantClient,
    collection_name: str, openai_api_key: str,
    filt=None, top_k: int = config.TOP_K,
) -> list:
    """
    Retrieves top_k * 2 chunks, then uses an LLM to extract only the
    sentences within each chunk that are relevant to the query.
    Produces cleaner, more focused context — at the cost of extra LLM calls.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore

    ContextualCompressionRetriever, LLMChainExtractor = _compression_retriever_components()
    lc_embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=openai_api_key)
    vectorstore   = QdrantVectorStore(
        client=qdrant,
        collection_name=collection_name,
        embedding=lc_embeddings,
        vector_name="dense",
    )
    llm        = ChatOpenAI(model=config.LLM_MODEL, api_key=openai_api_key, temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    retriever  = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": top_k * 2}),
    )
    docs = retriever.invoke(query)
 
    seen   = set()
    points = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
 
            class _FakePoint:
                payload = {**doc.metadata, "text": doc.page_content}
                score   = None
 
            points.append(_FakePoint())
 
    return points[:top_k]
 
 
# ── Cross-encoder reranking (your existing code) ──────────────────────────────
 
def rerank(query: str, points: list, cross_encoder: CrossEncoder,
           top_k: int = config.TOP_K) -> list:
    """
    Re-scores a candidate list with a cross-encoder and returns the top_k.
    Call this after dense_search or hybrid_search.
    """
    pairs  = [(query, p.payload.get("text", "")) for p in points]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)
    for score, point in ranked:
        point._rerank_score = float(score)
    return [p for _, p in ranked[:top_k]]
 
 
# ── Main retrieve() — unified entry point ─────────────────────────────────────
 
def retrieve(
    query:           str,
    strategy:        str,
    oai:             OpenAI,
    qdrant:          QdrantClient,
    collection_name: str,
    bm25_model:      SparseTextEmbedding = None,
    cross_encoder:   CrossEncoder        = None,
    openai_api_key:  str                 = None,
    top_k:           int                 = config.TOP_K,
    use_rerank:      bool                = False,
    category:        str                 = None,
    source:          str                 = None,
    region:          str                 = None,
    role_relevance:  str                 = None,
    answer_type:     str                 = None,
) -> dict:
    """
    Unified retrieval entry point. Pass strategy= to pick the method.
 
    Strategies:
        "dense"       — cosine similarity only
        "hybrid"      — dense + BM25 with RRF (requires bm25_model)
        "multi_query" — LLM query rewriting + merge (requires openai_api_key)
        "compression" — dense + LLM compression (requires openai_api_key)
 
    Add use_rerank=True to any strategy to apply cross-encoder reranking on top
    (requires cross_encoder).
 
    Usage:
        result = retrieve("What are FWA rights?", strategy="hybrid",
                          oai=oai, qdrant=qdrant,
                          collection_name="hr_rag_recursive",
                          bm25_model=bm25_model)
        chunks = result["retrieved_chunks"]
    """
    filter_map = {
        "category": category, "source": source, "region": region,
        "role_relevance": role_relevance, "answer_type": answer_type,
    }
    filt            = _build_filter(**filter_map)
    filters_applied = {k: v for k, v in filter_map.items() if v is not None}
 
    fetch_k = top_k * config.RERANK_FACTOR if use_rerank else top_k
 
    if strategy == "dense":
        points = dense_search(query, oai, qdrant, collection_name, filt, fetch_k)
    elif strategy == "hybrid":
        points = hybrid_search(query, oai, qdrant, bm25_model, collection_name, filt, fetch_k)
    elif strategy == "multi_query":
        points = multi_query_search(query, oai, qdrant, collection_name, openai_api_key, filt, fetch_k)
    elif strategy == "compression":
        points = compression_search(query, oai, qdrant, collection_name, openai_api_key, filt, fetch_k)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {config.RETRIEVAL_STRATEGIES}")
 
    if use_rerank and cross_encoder:
        points = rerank(query, points, cross_encoder, top_k)
 
    return _format_results(points, strategy, query, top_k, filters_applied)