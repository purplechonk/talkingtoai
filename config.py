# config.py
# Central configuration for the HR Wellness RAG project.
# All notebooks import from here — change a value once, it applies everywhere.
 
# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
LLM_MODEL       = "gpt-4o-mini"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
 
# ── Qdrant ────────────────────────────────────────────────────────────────────
# One collection per chunking strategy.
# e.g. hr_rag_recursive, hr_rag_semantic, hr_rag_header, hr_rag_parent_child
COLLECTION_PREFIX = "hr_rag"
 
# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNKING_STRATEGIES = ["recursive", "semantic", "header", "parent_child"]
 
CHUNK_SIZE        = 400   # tokens (used by recursive + parent_child)
CHUNK_OVERLAP     = 60
MIN_CHUNK_CHARS   = 150
PARENT_CHUNK_SIZE = 1200  # larger parent chunk for parent_child strategy
 
# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_STRATEGIES = ["dense", "hybrid", "multi_query", "compression"]
TOP_K         = 5
RERANK_FACTOR = 4   # fetch top_k * RERANK_FACTOR before reranking
 
# ── Metadata tagging ──────────────────────────────────────────────────────────
TAG_BATCH_SIZE   = 20
EMBED_BATCH_SIZE = 100
 
# ── Evaluation ────────────────────────────────────────────────────────────────
GT_PAIRS    = 50
RANDOM_SEED = 42
 
# ── Paths (Google Drive) ──────────────────────────────────────────────────────
DRIVE_BASE       = "/content/drive/MyDrive/hr_wellness_rag"
CORPUS_PATH      = f"{DRIVE_BASE}/scraped_corpus.json"
CLEAN_CHUNKS_DIR = f"{DRIVE_BASE}/chunks"   # one file per strategy, e.g. chunks/recursive.json
EMBEDDINGS_DIR   = f"{DRIVE_BASE}/embeddings"
GT_PAIRS_PATH    = f"{DRIVE_BASE}/gt_pairs.json"
RESULTS_PATH     = f"{DRIVE_BASE}/results/eval_results.csv"
 
# ── Filterable metadata fields ────────────────────────────────────────────────
INDEXED_FIELDS = ["source", "category", "region", "role_relevance", "answer_type"]
 
CATEGORIES = [
    "mental_health", "flexible_work", "harassment", "workplace_wellness",
    "stress_psychosocial", "health_data", "physical_wellness",
    "workplace_fairness", "policy", "wsh_statistics", "crisis_support",
    "employment_practices",
]