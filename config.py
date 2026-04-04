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
# e.g. hr_rag_recursive, hr_rag_semantic, hr_rag_sentence_window, hr_rag_parent_child, hr_rag_proposition
COLLECTION_PREFIX = "hr_rag"
 
# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNKING_STRATEGIES = ["recursive", "semantic", "sentence_window", "parent_child", "proposition"]

CHUNK_SIZE           = 400    # tokens (recursive, parent_child child, proposition base passages)
CHUNK_OVERLAP        = 60
MIN_CHUNK_CHARS      = 150
PARENT_CHUNK_SIZE    = 1200   # larger parent chunk for parent_child strategy
SENTENCE_WINDOW_SIZE = 3      # sentences before + after for sentence_window strategy
 
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
 
# ── Paths ─────────────────────────────────────────────────────────────────────
from pathlib import Path as _Path

_ROOT            = _Path(__file__).parent
SOURCES_PATH     = str(_ROOT / "data" / "sources.json")          # URL manifest (committed)
CORPUS_PATH      = str(_ROOT / "data" / "scraped_corpus.json")   # scraped text (git-ignored)
CLEAN_CHUNKS_DIR = str(_ROOT / "data" / "chunks")                # one JSON per strategy
EMBEDDINGS_DIR   = str(_ROOT / "data" / "embeddings")            # embedding cache per strategy
DOWNLOADS_DIR    = str(_ROOT / "data" / "downloads")             # raw PDF downloads
GT_PAIRS_PATH    = str(_ROOT / "data" / "gt_pairs.json")
RESULTS_PATH     = str(_ROOT / "results" / "eval_results.csv")
CHUNK_EVAL_PATH  = str(_ROOT / "results" / "chunk_eval_results.csv")
SYNTH_QUERIES_DIR = str(_ROOT / "data" / "synth_queries")
 
# ── Filterable metadata fields ────────────────────────────────────────────────
INDEXED_FIELDS = ["source", "category", "region", "role_relevance", "answer_type"]
 
CATEGORIES = [
    "mental_health", "flexible_work", "harassment", "workplace_wellness",
    "stress_psychosocial", "health_data", "physical_wellness",
    "workplace_fairness", "policy", "wsh_statistics", "crisis_support",
    "employment_practices",
]