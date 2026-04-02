# pipeline/embedder.py
# Metadata tagging, embedding, and Qdrant upsert.
# Extracted directly from retrieval_vectordb.ipynb — same logic, now importable.
 
import json
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseIndexParams, PointStruct, SparseVector,
)
from fastembed import SparseTextEmbedding
 
import sys
sys.path.append("..")
import config
 
 
# ── Metadata tagging (from retrieval_vectordb.ipynb Cell 5 & 6) ──────────────
 
EMPLOYEE_KW = {
    "leave", "maternity", "paternity", "resignation", "salary", "wages",
    "grievance", "complaint", "wellbeing", "well-being", "mental health",
    "stress", "burnout", "benefits", "sick leave", "annual leave",
    "discrimination", "harassment", "rights", "entitlement", "work-life",
}
MANAGER_KW = {
    "manager", "supervisor", "team lead", "manage", "leadership",
    "performance review", "appraisal", "feedback", "delegation",
    "coaching", "team management", "conflict resolution",
    "subordinate", "direct report", "line manager",
}
HR_KW = {
    "policy", "compliance", "onboarding", "offboarding", "recruitment",
    "hiring", "termination", "dismissal", "handbook",
    "payroll", "human resources", "employment act", "tripartite",
    "hr department", "benefits administration",
}
CATEGORY_ROLE_MAP = {
    "mental_health":        "employee",
    "workplace_wellness":   "employee",
    "physical_wellness":    "employee",
    "stress_psychosocial":  "employee",
    "flexible_work":        "all",
    "employment_practices": "hr",
    "workplace_fairness":   "hr",
    "harassment":           "hr",
    "policy":               "hr",
    "wsh_statistics":       "all",
    "health_data":          "all",
}
 
TAGGING_SYSTEM = (
    "You are a metadata tagger for an HR knowledge base.\n"
    "Given a numbered list of text chunks, return ONLY a valid JSON object "
    "with a single key \"results\" whose value is an array of objects, one per chunk.\n"
    "Each object must have exactly two keys:\n"
    "  \"answer_type\": one of: policy, procedure, statistics, guidance, rights, definition, example\n"
    "  \"topic_tags\" : array of 2-3 lowercase keyword strings (specific subtopics)\n"
    "\n"
    "Definitions:\n"
    "  policy      — describes an official rule or regulation\n"
    "  procedure   — step-by-step process or checklist\n"
    "  statistics  — contains data, figures, percentages, or survey results\n"
    "  guidance    — advisory recommendations or best practices\n"
    "  rights      — employee or employer entitlements\n"
    "  definition  — defines a concept or term\n"
    "  example     — illustrative case, scenario, or case study"
)
 
 
def assign_role_relevance(chunk: dict) -> str:
    text_lower   = chunk["text"].lower()
    category     = chunk.get("category", "")
    manager_hits = sum(1 for kw in MANAGER_KW if kw in text_lower)
    hr_hits      = sum(1 for kw in HR_KW      if kw in text_lower)
    emp_hits     = sum(1 for kw in EMPLOYEE_KW if kw in text_lower)
 
    if manager_hits >= 2:
        return "manager"
    if hr_hits >= 2:
        return "hr"
    if emp_hits >= 2:
        return "employee"
    return CATEGORY_ROLE_MAP.get(category, "all")
 
 
def _build_tagging_prompt(batch: list) -> str:
    lines = []
    for i, chunk in enumerate(batch):
        preview = chunk["text"][:300].replace("\n", " ")
        lines.append(f"[{i}] SOURCE={chunk['source']} CATEGORY={chunk['category']}\nTEXT: {preview}")
    return "\n\n".join(lines)
 
 
def tag_batch(batch: list, oai: OpenAI, retries: int = 3) -> list:
    prompt = _build_tagging_prompt(batch)
    for attempt in range(retries):
        try:
            response = oai.chat.completions.create(
                model=config.LLM_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": TAGGING_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw    = response.choices[0].message.content.strip()
            raw    = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)
            return parsed["results"]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [WARN] tagging failed after {retries} attempts: {e}")
                return [{"answer_type": "guidance", "topic_tags": []} for _ in batch]
            time.sleep(2 ** attempt)
 
 
def tag_all_chunks(chunks: list, oai: OpenAI) -> list:
    """Add role_relevance, answer_type, topic_tags to every chunk."""
    # role_relevance is rule-based (fast, free)
    for c in chunks:
        c["role_relevance"] = assign_role_relevance(c)
 
    # answer_type + topic_tags via LLM in batches
    for i in tqdm(range(0, len(chunks), config.TAG_BATCH_SIZE), desc="Tagging"):
        batch   = chunks[i : i + config.TAG_BATCH_SIZE]
        tags    = tag_batch(batch, oai)
        for chunk, tag in zip(batch, tags):
            chunk["answer_type"] = tag.get("answer_type", "guidance")
            chunk["topic_tags"]  = tag.get("topic_tags", [])
        time.sleep(0.3)
 
    return chunks
 
 
# ── Embedding ─────────────────────────────────────────────────────────────────
 
def embed_texts(texts: list, oai: OpenAI) -> list:
    response = oai.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]
 
 
def build_embedding_cache(chunks: list, oai: OpenAI, cache_path: str = None) -> dict:
    """
    Embed all chunks, with an optional on-disk cache so you don't re-embed
    on re-runs. Cache is keyed by chunk_id.
    """
    cache = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache):,} cached embeddings from {cache_path}")
 
    to_embed = [c for c in chunks if c["chunk_id"] not in cache]
    print(f"Chunks to embed: {len(to_embed):,}")
 
    for i in tqdm(range(0, len(to_embed), config.EMBED_BATCH_SIZE), desc="Embedding"):
        batch = to_embed[i : i + config.EMBED_BATCH_SIZE]
        # For parent_child strategy, embed the child text only
        texts = [c["text"] for c in batch]
        vecs  = embed_texts(texts, oai)
        for chunk, vec in zip(batch, vecs):
            cache[chunk["chunk_id"]] = vec
 
        # Save every 5 batches
        if cache_path and (i // config.EMBED_BATCH_SIZE + 1) % 5 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
 
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
 
    return cache
 
 
# ── Qdrant collection setup + upsert ─────────────────────────────────────────
 
def create_collection(qdrant: QdrantClient, collection_name: str, recreate: bool = False):
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection_name in existing:
        if recreate:
            qdrant.delete_collection(collection_name)
            print(f"Deleted existing '{collection_name}'.")
        else:
            print(f"Collection '{collection_name}' already exists — skipping creation.")
            return
 
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    for field in config.INDEXED_FIELDS:
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    print(f"Created collection '{collection_name}' with dense + sparse vectors.")
 
 
def upsert_chunks(
    chunks: list,
    embedding_cache: dict,
    qdrant: QdrantClient,
    bm25_model: SparseTextEmbedding,
    collection_name: str,
    batch_size: int = 100,
):
    """Upsert all chunks into a Qdrant collection."""
    # Pre-generate all sparse vectors
    all_texts          = [c["text"] for c in chunks]
    sparse_embeddings  = list(bm25_model.embed(all_texts))
 
    uploaded = 0
    skipped  = 0
 
    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading"):
        batch        = chunks[i : i + batch_size]
        sparse_batch = sparse_embeddings[i : i + batch_size]
        points       = []
 
        for chunk, sp_emb in zip(batch, sparse_batch):
            cid = chunk["chunk_id"]
            if cid not in embedding_cache:
                skipped += 1
                continue
 
            # Build payload — include parent_text for parent_child strategy
            payload = {
                "chunk_id":       cid,
                "text":           chunk["text"],
                "token_count":    chunk["token_count"],
                "source":         chunk.get("source"),
                "category":       chunk.get("category"),
                "region":         chunk.get("region"),
                "type":           chunk.get("type"),
                "title":          chunk.get("title"),
                "url":            chunk.get("url"),
                "scraped_at":     chunk.get("scraped_at"),
                "answer_type":    chunk.get("answer_type", "guidance"),
                "role_relevance": chunk.get("role_relevance", "all"),
                "topic_tags":     chunk.get("topic_tags", []),
                "strategy":       chunk.get("strategy"),
                "parent_id":      chunk.get("parent_id"),
                "parent_text":    chunk.get("parent_text"),  # None for non-parent_child
            }
 
            points.append(PointStruct(
                id=cid,
                vector={
                    "dense":  embedding_cache[cid],
                    "sparse": SparseVector(
                        indices=sp_emb.indices.tolist(),
                        values=sp_emb.values.tolist(),
                    ),
                },
                payload=payload,
            ))
 
        if points:
            qdrant.upsert(collection_name=collection_name, points=points)
            uploaded += len(points)
 
    print(f"Done. Uploaded: {uploaded:,}  |  Skipped (no embedding): {skipped}")
 
 
# ── Main entry point ──────────────────────────────────────────────────────────
 
def build_index(
    chunks: list,
    strategy: str,
    oai: OpenAI,
    qdrant: QdrantClient,
    bm25_model: SparseTextEmbedding,
    recreate: bool = False,
):
    """
    Full pipeline: tag → embed → upsert.
    Call this once per chunking strategy.
 
    Usage in notebook:
        from pipeline.embedder import build_index
        build_index(chunks, strategy="recursive", oai=oai, qdrant=qdrant, bm25_model=bm25_model)
    """
    collection_name = f"{config.COLLECTION_PREFIX}_{strategy}"
    cache_path      = f"{config.EMBEDDINGS_DIR}/{strategy}_embeddings.json"
 
    print(f"\n{'='*60}")
    print(f"Building index: {collection_name}")
    print(f"{'='*60}")
 
    # Skip parent chunks from tagging/embedding — only children go into Qdrant
    indexable = [c for c in chunks if c.get("strategy") != "parent"]
 
    print(f"\n[1/3] Tagging {len(indexable):,} chunks...")
    tag_all_chunks(indexable, oai)
 
    print(f"\n[2/3] Embedding...")
    Path(config.EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
    embedding_cache = build_embedding_cache(indexable, oai, cache_path=cache_path)
 
    print(f"\n[3/3] Upserting to Qdrant collection '{collection_name}'...")
    create_collection(qdrant, collection_name, recreate=recreate)
    upsert_chunks(indexable, embedding_cache, qdrant, bm25_model, collection_name)
 
    info = qdrant.get_collection(collection_name)
    print(f"\nCollection '{collection_name}' ready — {info.points_count:,} points.")