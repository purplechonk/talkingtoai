# pipeline/rag.py
# RAG generation layer — extracted from rag.ipynb Cells 12-13.
# Takes retrieved chunks, builds context, calls the LLM.
 
from openai import OpenAI
from pipeline.retriever import retrieve
 
import sys
sys.path.append("..")
import config
 
HR_SYSTEM_PROMPT = """\
You are an HR assistant specialising in Singapore workplace policies and employee well-being.
Your knowledge comes from official Singapore sources including MOM (Ministry of Manpower), TAFEP,
HPB, MOH, WSH Council, NCSS, ILO, AWARE, and SOS.
 
Answer the question using ONLY the context provided below. Be specific and cite the source where possible.
If the context does not contain enough information to answer, say exactly:
"The provided sources do not contain sufficient information to answer this question."
"""
 
 
def build_context(chunks: list) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.
    For parent_child chunks, uses context_text (the larger parent) instead
    of text (the small child) to give the LLM more surrounding information.
    """
    parts = []
    for c in chunks:
        meta = (
            f"[Source: {c.get('source')} | "
            f"Category: {c.get('category')} | "
            f"Type: {c.get('answer_type')}]"
        )
        # context_text is the parent chunk for parent_child strategy,
        # falls back to text for all other strategies
        body = c.get("context_text") or c.get("text", "")
        parts.append(f"{meta}\n{body}")
    return "\n\n---\n\n".join(parts)
 
 
def rag(
    query:           str,
    strategy:        str,
    chunking:        str,
    oai:             OpenAI,
    qdrant,
    bm25_model=None,
    cross_encoder=None,
    openai_api_key:  str  = None,
    top_k:           int  = config.TOP_K,
    use_rerank:      bool = False,
    **filters,
) -> dict:
    """
    Full RAG pipeline: retrieve → build context → generate answer.
 
    Args:
        query          : the user's question
        strategy       : retrieval strategy ("dense", "hybrid", "multi_query", "compression")
        chunking       : chunking strategy used to build the index ("recursive", "semantic", etc.)
                         determines which Qdrant collection to query
        oai            : OpenAI client
        qdrant         : QdrantClient
        bm25_model     : required for "hybrid" strategy
        cross_encoder  : required when use_rerank=True
        openai_api_key : required for "multi_query" and "compression" strategies
        top_k          : number of chunks to retrieve
        use_rerank     : apply cross-encoder reranking on top of retrieval
        **filters      : optional metadata filters (category, source, region, role_relevance, answer_type)
 
    Returns:
        {
            "query":            str,
            "answer":           str,
            "contexts":         list[str],   # raw chunk texts (for RAGAS)
            "chunks":           list[dict],  # full chunk metadata
            "retrieval_strategy": str,
            "chunking_strategy":  str,
        }
 
    Usage:
        result = rag(
            query="What are FWA rights for employees?",
            strategy="hybrid",
            chunking="recursive",
            oai=oai, qdrant=qdrant, bm25_model=bm25_model,
            category="flexible_work",
        )
        print(result["answer"])
    """
    collection_name = f"{config.COLLECTION_PREFIX}_{chunking}"
 
    result = retrieve(
        query=query,
        strategy=strategy,
        oai=oai,
        qdrant=qdrant,
        collection_name=collection_name,
        bm25_model=bm25_model,
        cross_encoder=cross_encoder,
        openai_api_key=openai_api_key,
        top_k=top_k,
        use_rerank=use_rerank,
        **filters,
    )
 
    chunks   = result["retrieved_chunks"]
    context  = build_context(chunks)
    # RAGAS expects a list of strings — use the actual retrieved text, not parent
    contexts = [c["text"] for c in chunks]
 
    prompt = f"{HR_SYSTEM_PROMPT}\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
 
    response = oai.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
 
    return {
        "query":               query,
        "answer":              answer,
        "contexts":            contexts,
        "chunks":              chunks,
        "retrieval_strategy":  strategy,
        "chunking_strategy":   chunking,
    }