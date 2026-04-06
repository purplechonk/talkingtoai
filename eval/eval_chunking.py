# eval/eval_chunking.py
# Chunking strategy evaluation — intrinsic quality + retrieval metrics.
# Picks the best chunking strategy before retrieval-method comparison (notebook 03).

import json
import math
import os
import re
import time
import random
import numpy as np
import pandas as pd
import tiktoken
from collections import defaultdict
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ── Data loading ──────────────────────────────────────────────────────────────

def detect_available_collections(qdrant: QdrantClient) -> list[str]:
    """Return strategy names whose Qdrant collection exists and has points."""
    prefix = config.COLLECTION_PREFIX + "_"
    available = []
    for col in qdrant.get_collections().collections:
        if col.name.startswith(prefix):
            strategy = col.name[len(prefix):]
            if strategy in config.CHUNKING_STRATEGIES:
                info = qdrant.get_collection(col.name)
                if info.points_count and info.points_count > 0:
                    available.append(strategy)
    return sorted(available, key=lambda s: config.CHUNKING_STRATEGIES.index(s))


def load_chunks_from_qdrant(qdrant: QdrantClient, collection_name: str) -> list[dict]:
    """Scroll all payloads from a Qdrant collection."""
    all_points = []
    offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            offset=offset,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset
    return [p.payload for p in all_points]


def load_chunks_from_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Tier 1: Intrinsic metrics ────────────────────────────────────────────────

def compute_size_compliance(chunks: list[dict],
                            min_tokens: int = 100,
                            max_tokens: int = 500) -> float:
    """Fraction of chunks within [min_tokens, max_tokens]."""
    if not chunks:
        return 0.0
    in_range = sum(
        1 for c in chunks
        if min_tokens <= c.get("token_count", _count_tokens(c.get("text", ""))) <= max_tokens
    )
    return in_range / len(chunks)


def compute_token_stats(chunks: list[dict]) -> dict:
    """Return mean, median, std, min, max token counts."""
    counts = [
        c.get("token_count", _count_tokens(c.get("text", "")))
        for c in chunks
    ]
    if not counts:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "total_chunks": 0}
    arr = np.array(counts, dtype=float)
    return {
        "mean":         round(float(np.mean(arr)), 1),
        "median":       round(float(np.median(arr)), 1),
        "std":          round(float(np.std(arr)), 1),
        "min":          int(np.min(arr)),
        "max":          int(np.max(arr)),
        "total_chunks": len(counts),
    }


def compute_intrachunk_cohesion(chunks: list[dict], st_model,
                                 sample_n: int = 300,
                                 seed: int = config.RANDOM_SEED) -> float | None:
    """
    Mean cosine similarity between a chunk's sentences and the whole-chunk
    embedding.  Uses a sentence-transformers model (runs locally on CPU).
    Samples up to sample_n chunks for speed.
    """
    _SENT_RE = re.compile(r"(?<=[.!?])\s+")
    rng = random.Random(seed)
    pool = chunks if len(chunks) <= sample_n else rng.sample(chunks, sample_n)

    chunk_texts = [c.get("text", "") for c in pool]
    chunk_sentences: list[list[str]] = []
    for text in chunk_texts:
        sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
        chunk_sentences.append(sents if len(sents) >= 2 else [text])

    flat_sentences = [s for sents in chunk_sentences for s in sents]
    if not flat_sentences:
        return None

    chunk_embs = st_model.encode(chunk_texts, normalize_embeddings=True,
                                  show_progress_bar=False)
    sent_embs = st_model.encode(flat_sentences, normalize_embeddings=True,
                                 show_progress_bar=False)

    scores = []
    idx = 0
    for i, sents in enumerate(chunk_sentences):
        n = len(sents)
        if n < 2:
            idx += n
            continue
        s_embs = sent_embs[idx: idx + n]
        sims = np.dot(s_embs, chunk_embs[i])
        scores.append(float(np.mean(sims)))
        idx += n

    return round(float(np.mean(scores)), 4) if scores else None


def compute_interchunk_dissimilarity(chunks: list[dict], st_model,
                                      window_size: int = 5,
                                      sample_n: int = 500,
                                      seed: int = config.RANDOM_SEED) -> float | None:
    """
    Weighted cosine dissimilarity between neighbouring chunks in a sliding
    window.  Higher = better topic separation.
    """
    rng = random.Random(seed)
    if len(chunks) < 2:
        return None

    if len(chunks) > sample_n:
        start = rng.randint(0, len(chunks) - sample_n)
        subset = chunks[start: start + sample_n]
    else:
        subset = chunks

    texts = [c.get("text", "") for c in subset]
    embs = st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    counts = [c.get("token_count", _count_tokens(t)) for c, t in zip(subset, texts)]

    total_sim, total_wt = 0.0, 0.0
    for i in range(len(embs)):
        for j in range(i + 1, min(i + window_size + 1, len(embs))):
            sim = float(np.dot(embs[i], embs[j]))
            wt = counts[i] * counts[j]
            total_sim += sim * wt
            total_wt += wt

    if total_wt == 0:
        return None
    dissim = 1.0 - (total_sim / total_wt)
    return round(float(np.clip(dissim, 0.0, 1.0)), 4)


def compute_overlap_ratio(chunks: list[dict]) -> float | None:
    """
    For sentence_window chunks: fraction of window_text shared between
    consecutive chunks.  Returns None for strategies without window_text.
    """
    windows = [c.get("window_text") for c in chunks if c.get("window_text")]
    if len(windows) < 2:
        return None

    overlaps = []
    for a, b in zip(windows, windows[1:]):
        set_a = set(a.split())
        set_b = set(b.split())
        union = set_a | set_b
        if union:
            overlaps.append(len(set_a & set_b) / len(union))
    return round(float(np.mean(overlaps)), 4) if overlaps else None


def compute_context_completeness(chunks: list[dict]) -> float | None:
    """
    For parent_child / proposition: fraction of chunks whose parent_text or
    source_passage fully contains the child/proposition text.
    """
    checked, contained = 0, 0
    for c in chunks:
        text = c.get("text", "")
        context = c.get("parent_text") or c.get("source_passage")
        if not context:
            continue
        checked += 1
        if text.strip() in context:
            contained += 1
    if checked == 0:
        return None
    return round(contained / checked, 4)


def compute_category_distribution(chunks: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = defaultdict(int)
    for c in chunks:
        dist[c.get("category", "unknown")] += 1
    return dict(sorted(dist.items()))


def run_intrinsic_eval(
    strategies: list[str],
    qdrant: QdrantClient,
    st_model,
) -> pd.DataFrame:
    """Run all Tier 1 metrics for each strategy, return a comparison DataFrame."""
    rows = []
    for strategy in strategies:
        collection = f"{config.COLLECTION_PREFIX}_{strategy}"
        print(f"\n[Tier 1] {strategy} — loading from '{collection}'...")
        chunks = load_chunks_from_qdrant(qdrant, collection)
        indexable = [c for c in chunks if c.get("strategy") != "parent"]
        print(f"  {len(indexable):,} indexable chunks loaded.")

        stats = compute_token_stats(indexable)
        row = {
            "strategy":         strategy,
            "total_chunks":     stats["total_chunks"],
            "token_mean":       stats["mean"],
            "token_median":     stats["median"],
            "token_std":        stats["std"],
            "token_min":        stats["min"],
            "token_max":        stats["max"],
            "size_compliance":  round(compute_size_compliance(indexable), 4),
        }

        print("  Computing intrachunk cohesion...")
        row["icc"] = compute_intrachunk_cohesion(indexable, st_model)

        print("  Computing inter-chunk dissimilarity...")
        row["dissimilarity"] = compute_interchunk_dissimilarity(indexable, st_model)

        row["overlap_ratio"]        = compute_overlap_ratio(indexable)
        row["context_completeness"] = compute_context_completeness(indexable)
        row["categories_covered"]   = len(compute_category_distribution(indexable))

        rows.append(row)
        print(f"  Done: ICC={row['icc']}  dissim={row['dissimilarity']}  "
              f"size_comp={row['size_compliance']}")

    return pd.DataFrame(rows)


# ── Tier 2: Synthetic query generation ────────────────────────────────────────

_SYNTH_SYSTEM = """\
You are generating a retrieval evaluation dataset.
Given a text chunk and metadata, generate ONE specific question that this
passage uniquely answers.  The question should be natural — something a real
employee, HR professional, or manager would ask.

Respond with ONLY valid JSON: {"question": "..."}"""


def _build_synth_prompt(chunk: dict) -> str:
    return (
        f"Source: {chunk.get('source')} — {chunk.get('title')}\n"
        f"Category: {chunk.get('category')}\n"
        f"Answer type: {chunk.get('answer_type')}\n\n"
        f"Chunk text:\n\"\"\"\n{chunk.get('text', '')[:1500]}\n\"\"\"\n\n"
        "Generate the JSON now."
    )


def _generate_one_query(chunk: dict, oai: OpenAI) -> dict | None:
    try:
        resp = oai.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": _SYNTH_SYSTEM},
                {"role": "user",   "content": _build_synth_prompt(chunk)},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        return {
            "question": parsed["question"],
            "chunk_id": chunk.get("chunk_id"),
            "category": chunk.get("category"),
            "source":   chunk.get("source"),
        }
    except Exception as e:
        print(f"    [WARN] query gen failed: {e}")
        return None


def _sample_evenly(chunks: list, n: int, seed: int = config.RANDOM_SEED) -> list:
    by_cat: dict[str, list] = defaultdict(list)
    for c in chunks:
        by_cat[c.get("category", "unknown")].append(c)

    rng = random.Random(seed)
    sampled: list = []
    cats = list(by_cat.keys())
    per_cat = max(1, n // len(cats))

    for cat in cats:
        pool = by_cat[cat]
        sampled.extend(rng.sample(pool, min(per_cat, len(pool))))

    remaining = [c for c in chunks if c not in sampled]
    rng.shuffle(remaining)
    sampled.extend(remaining[: max(0, n - len(sampled))])
    return sampled[:n]


def generate_synthetic_queries(
    chunks: list[dict],
    oai: OpenAI,
    n: int = 100,
    save_path: str | None = None,
) -> list[dict]:
    """Generate n synthetic queries from sampled chunks.  Saves to JSON."""
    sampled = _sample_evenly(chunks, n)
    print(f"  Generating {len(sampled)} synthetic queries...")
    queries: list[dict] = []
    for i, chunk in enumerate(sampled, 1):
        print(f"  [{i:03d}/{len(sampled)}] {chunk.get('category', '?')[:20]}...", end=" ", flush=True)
        q = _generate_one_query(chunk, oai)
        if q:
            queries.append(q)
            print("ok")
        else:
            print("SKIP")
        time.sleep(0.25)

    print(f"  Generated {len(queries)} queries.")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {save_path}")
    return queries


def load_synthetic_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Tier 2: Retrieval metrics ─────────────────────────────────────────────────

def _dense_search_ids_scores(
    query: str, oai: OpenAI, qdrant: QdrantClient,
    collection_name: str, top_k: int,
) -> list[dict]:
    """Return list of {chunk_id, score, text, context_text, token_count}."""
    vec = oai.embeddings.create(model=config.EMBEDDING_MODEL, input=[query]).data[0].embedding
    results = qdrant.query_points(
        collection_name=collection_name,
        query=vec,
        using="dense",
        limit=top_k,
        with_payload=True,
    ).points
    out = []
    for r in results:
        p = r.payload
        out.append({
            "chunk_id":    p.get("chunk_id"),
            "score":       float(r.score) if r.score is not None else None,
            "text":        p.get("text", ""),
            "context_text": (p.get("parent_text") or p.get("window_text")
                             or p.get("source_passage") or p.get("text", "")),
            "token_count": p.get("token_count", _count_tokens(p.get("text", ""))),
        })
    return out


def compute_hit_rate(retrieved_ids: list[list[str]], gt_ids: list[str], k: int) -> float:
    """Fraction of queries where the GT chunk_id appears in top-k."""
    hits = 0
    for ret, gt in zip(retrieved_ids, gt_ids):
        if gt in ret[:k]:
            hits += 1
    return round(hits / len(gt_ids), 4) if gt_ids else 0.0


def compute_mrr(retrieved_ids: list[list[str]], gt_ids: list[str]) -> float:
    """Mean Reciprocal Rank."""
    rr_sum = 0.0
    for ret, gt in zip(retrieved_ids, gt_ids):
        for rank, cid in enumerate(ret, 1):
            if cid == gt:
                rr_sum += 1.0 / rank
                break
    return round(rr_sum / len(gt_ids), 4) if gt_ids else 0.0


def compute_precision_at_k(retrieved_ids: list[list[str]], gt_ids: list[str], k: int) -> float:
    """Precision @ K (with 1 relevant doc per query, equals hit_rate / k)."""
    prec_sum = 0.0
    for ret, gt in zip(retrieved_ids, gt_ids):
        relevant_in_k = sum(1 for cid in ret[:k] if cid == gt)
        prec_sum += relevant_in_k / k
    return round(prec_sum / len(gt_ids), 4) if gt_ids else 0.0


def compute_ndcg_at_k(retrieved_ids: list[list[str]], gt_ids: list[str], k: int) -> float:
    """NDCG @ K with binary relevance (1 relevant doc per query)."""
    ndcg_sum = 0.0
    idcg = 1.0  # ideal: relevant doc at rank 1 → 1/log2(2) = 1.0
    for ret, gt in zip(retrieved_ids, gt_ids):
        dcg = 0.0
        for rank, cid in enumerate(ret[:k], 1):
            if cid == gt:
                dcg += 1.0 / math.log2(rank + 1)
        ndcg_sum += dcg / idcg
    return round(ndcg_sum / len(gt_ids), 4) if gt_ids else 0.0


# ── Tier 2b: Score analysis ──────────────────────────────────────────────────

def compute_score_stats(all_results: list[list[dict]]) -> dict:
    """
    From a list of per-query result lists, compute score distribution stats.
    Each inner list is the output of _dense_search_ids_scores.
    """
    top1_scores = []
    topk_scores = []
    gaps = []

    for results in all_results:
        scores = [r["score"] for r in results if r["score"] is not None]
        if not scores:
            continue
        top1_scores.append(scores[0])
        topk_scores.extend(scores)
        if len(scores) >= 2:
            gaps.append(scores[0] - scores[-1])

    return {
        "mean_top1_score": round(float(np.mean(top1_scores)), 4) if top1_scores else None,
        "std_top1_score":  round(float(np.std(top1_scores)), 4) if top1_scores else None,
        "mean_score_gap":  round(float(np.mean(gaps)), 4) if gaps else None,
        "median_all_scores": round(float(np.median(topk_scores)), 4) if topk_scores else None,
        "all_top1_scores": top1_scores,
        "all_topk_scores": topk_scores,
    }


def compute_avg_retrieved_tokens(all_results: list[list[dict]]) -> dict:
    """Mean total tokens of context_text across top-K per query."""
    text_totals = []
    context_totals = []
    for results in all_results:
        text_totals.append(sum(r["token_count"] for r in results))
        context_totals.append(sum(_count_tokens(r["context_text"]) for r in results))
    return {
        "avg_retrieved_tokens":     round(float(np.mean(text_totals)), 1) if text_totals else 0,
        "avg_context_tokens":       round(float(np.mean(context_totals)), 1) if context_totals else 0,
    }


# ── Tier 2: Orchestrated retrieval eval ──────────────────────────────────────

def run_retrieval_eval(
    queries: list[dict],
    strategy: str,
    oai: OpenAI,
    qdrant: QdrantClient,
    top_k: int = config.TOP_K,
) -> dict:
    """
    Run dense retrieval for each synthetic query against the strategy's
    collection.  Returns a dict with all Tier 2 + 2b metrics.
    """
    collection = f"{config.COLLECTION_PREFIX}_{strategy}"
    gt_ids: list[str] = []
    retrieved_ids: list[list[str]] = []
    all_results: list[list[dict]] = []

    for i, q in enumerate(queries, 1):
        if (i % 20 == 0) or i == 1:
            print(f"    Retrieving [{i}/{len(queries)}]...")
        results = _dense_search_ids_scores(q["question"], oai, qdrant, collection, top_k)
        gt_ids.append(q["chunk_id"])
        retrieved_ids.append([r["chunk_id"] for r in results])
        all_results.append(results)

    metrics = {
        "strategy":       strategy,
        "hit_rate_at_5":  compute_hit_rate(retrieved_ids, gt_ids, k=5),
        "mrr":            compute_mrr(retrieved_ids, gt_ids),
        "precision_at_5": compute_precision_at_k(retrieved_ids, gt_ids, k=5),
        "ndcg_at_5":      compute_ndcg_at_k(retrieved_ids, gt_ids, k=5),
    }

    score_stats = compute_score_stats(all_results)
    metrics["mean_top1_score"] = score_stats["mean_top1_score"]
    metrics["mean_score_gap"]  = score_stats["mean_score_gap"]

    token_stats = compute_avg_retrieved_tokens(all_results)
    metrics["avg_retrieved_tokens"] = token_stats["avg_retrieved_tokens"]
    metrics["avg_context_tokens"]   = token_stats["avg_context_tokens"]

    # Stash raw data for visualisations in the notebook
    metrics["_all_top1_scores"] = score_stats["all_top1_scores"]
    metrics["_all_topk_scores"] = score_stats["all_topk_scores"]

    return metrics


# ── Full orchestration ────────────────────────────────────────────────────────

def run_full_chunk_eval(
    strategies: list[str],
    oai: OpenAI,
    qdrant: QdrantClient,
    st_model,
    n_queries: int = 100,
    top_k: int = config.TOP_K,
    save_path: str | None = config.CHUNK_EVAL_PATH,
) -> pd.DataFrame:
    """
    Run Tier 1 (intrinsic) + Tier 2 (retrieval) for all strategies.
    Returns a merged DataFrame and saves to CSV.
    """
    print("=" * 60)
    print("CHUNKING STRATEGY EVALUATION")
    print("=" * 60)

    # Tier 1
    intrinsic_df = run_intrinsic_eval(strategies, qdrant, st_model)

    # Tier 2: generate or load synthetic queries per strategy
    retrieval_rows = []
    for strategy in strategies:
        collection = f"{config.COLLECTION_PREFIX}_{strategy}"
        query_path = os.path.join(config.SYNTH_QUERIES_DIR, f"{strategy}.json")

        if os.path.exists(query_path):
            print(f"\n[Tier 2] {strategy} — loading cached queries from {query_path}")
            queries = load_synthetic_queries(query_path)
        else:
            print(f"\n[Tier 2] {strategy} — generating synthetic queries...")
            chunks = load_chunks_from_qdrant(qdrant, collection)
            indexable = [c for c in chunks if c.get("strategy") != "parent"]
            queries = generate_synthetic_queries(
                indexable, oai, n=n_queries, save_path=query_path,
            )

        print(f"  Running retrieval eval ({len(queries)} queries, top_k={top_k})...")
        result = run_retrieval_eval(queries, strategy, oai, qdrant, top_k)
        retrieval_rows.append(result)

    retrieval_df = pd.DataFrame(retrieval_rows)

    # Drop raw score lists before merge
    vis_cols = [c for c in retrieval_df.columns if c.startswith("_")]
    retrieval_clean = retrieval_df.drop(columns=vis_cols, errors="ignore")

    merged = intrinsic_df.merge(retrieval_clean, on="strategy", how="outer")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")

    return merged, retrieval_df


def print_chunk_eval_summary(df: pd.DataFrame):
    """Print a readable summary table and recommend the best strategy."""
    print("\n" + "=" * 70)
    print("CHUNKING STRATEGY EVALUATION SUMMARY")
    print("=" * 70)

    display_cols = [c for c in [
        "strategy", "total_chunks",
        "hit_rate_at_5", "mrr", "ndcg_at_5",
        "icc", "size_compliance", "dissimilarity",
        "mean_top1_score", "mean_score_gap",
        "avg_context_tokens",
    ] if c in df.columns]

    summary = df[display_cols].copy()
    for c in summary.columns:
        if summary[c].dtype == float:
            summary[c] = summary[c].round(4)

    print(summary.to_string(index=False))
    print("=" * 70)

    ranking_cols = ["hit_rate_at_5", "mrr", "ndcg_at_5", "icc", "size_compliance"]
    available = [c for c in ranking_cols if c in df.columns and df[c].notna().any()]
    if available:
        weights = {"hit_rate_at_5": 0.35, "mrr": 0.25, "ndcg_at_5": 0.15,
                    "icc": 0.15, "size_compliance": 0.10}
        df = df.copy()
        df["composite"] = sum(
            df[c].fillna(0) * weights.get(c, 0) for c in available
        )
        best = df.loc[df["composite"].idxmax()]
        print(f"\nRecommended strategy: {best['strategy']}")
        print(f"  Composite score : {best['composite']:.4f}")
        for c in available:
            print(f"  {c:20s}: {best[c]:.4f}")
    print("=" * 70)
