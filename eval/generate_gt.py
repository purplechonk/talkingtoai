# eval/generate_gt.py
# Ground truth QA pair generation — extracted from rag.ipynb Cells 18-21.
# Run once; saves gt_pairs.json to Drive. Reuse across all eval runs.
 
import json
import time
import random
from collections import defaultdict
from openai import OpenAI
from qdrant_client import QdrantClient
 
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import config
 
ANSWER_TYPE_HINTS = {
    "statistics": "Ask a specific factual question about a number, percentage, rate, or statistic.",
    "guidance":   "Ask a practical 'how should' or 'what can' question the text directly answers.",
    "procedure":  "Ask a step-by-step or process question the text explains.",
    "policy":     "Ask about a rule, requirement, or policy position described in the text.",
    "example":    "Ask for a concrete example or illustration of something described.",
    "definition": "Ask what a specific term or concept means, as defined in the text.",
    "rights":     "Ask about an entitlement, right, or protection described in the text.",
}
 
SYSTEM_PROMPT = """\
You are building a RAGAS evaluation dataset for a workplace health and
employment RAG system focused on Singapore.
 
Given a text chunk and its metadata, generate ONE question–answer pair where:
  1. The question reflects what a real employee, HR professional, or manager might genuinely ask.
  2. The reference answer is fully grounded in the provided chunk text — no external knowledge.
  3. The reference answer is concise but complete (2–5 sentences).
 
Respond with ONLY valid JSON in this exact format (no markdown, no preamble):
{
  "question": "...",
  "reference": "..."
}"""
 
 
# ── Adversarial / failure-case eval set (hand-curated, from rag.ipynb) ────────
 
FAILURE_CASES = [
    {
        "query":   "What are the three types of Flexible Work Arrangements under Singapore's TG-FWAR?",
        "filters": {"category": "flexible_work"},
        "case":    "standard",
    },
    {
        "query":   "What should employers do to support employee mental well-being at the organisational level?",
        "filters": {"category": "mental_health", "source": "MOM"},
        "case":    "standard",
    },
    {
        "query":   "What are the five principles of Fair Employment Practices in Singapore?",
        "filters": {"category": "workplace_fairness", "source": "TAFEP"},
        "case":    "standard",
    },
    {
        "query":   "What counts as workplace harassment and what values should organisations adopt to prevent it?",
        "filters": {"category": "harassment"},
        "case":    "standard",
    },
    {
        "query":   "What crisis support services are available in Singapore for employees experiencing mental health issues?",
        "filters": {"category": "crisis_support"},
        "case":    "standard",
    },
    {
        "query":   "What are the main causes of work stress according to ILO guidelines?",
        "filters": {"category": "stress_psychosocial", "source": "ILO"},
        "case":    "standard",
    },
    # Out-of-domain
    {
        "query":   "What is the employer CPF contribution rate for workers above 55 in Singapore?",
        "filters": {},
        "case":    "out_of_domain",
    },
    # Source mismatch
    {
        "query":   "What does MOM recommend for managing employees with mental health conditions?",
        "filters": {"category": "mental_health", "source": "SHRM"},
        "case":    "source_mismatch",
    },
    # Ambiguous multi-category
    {
        "query":   "How should managers deal with a stressed employee who is also showing signs of burnout?",
        "filters": {"role_relevance": "manager"},
        "case":    "ambiguous_multi_category",
    },
    # Over-filtered
    {
        "query":   "What rights do employees have when requesting flexible work arrangements?",
        "filters": {"category": "flexible_work", "role_relevance": "hr", "answer_type": "rights"},
        "case":    "over_filtered",
    },
]
 
 
def _build_user_message(chunk: dict) -> str:
    hint = ANSWER_TYPE_HINTS.get(chunk.get("answer_type", "guidance"), ANSWER_TYPE_HINTS["guidance"])
    role = chunk.get("role_relevance", "all")
    tags = ", ".join(chunk.get("topic_tags") or []) or "none"
 
    return f"""\
Chunk metadata:
- Category      : {chunk.get("category")}
- Answer type   : {chunk.get("answer_type")}
- Role relevance: {role}
- Topic tags    : {tags}
- Source        : {chunk.get("source")} — {chunk.get("title")}
 
Question guidance: {hint}
Frame the question from the perspective of: {role}.
 
Chunk text:
\"\"\"
{chunk["text"][:1500]}
\"\"\"
 
Generate the JSON now."""
 
 
def _call_openai(chunk: dict, oai: OpenAI) -> dict | None:
    try:
        response = oai.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(chunk)},
            ],
        )
        raw    = response.choices[0].message.content.strip()
        raw    = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        return {
            "user_input": parsed["question"],
            "reference":  parsed["reference"],
            "response":   None,      # filled in during eval
            "retrieved_contexts": [],
            "chunk_id":   chunk.get("chunk_id"),
            "category":   chunk.get("category"),
            "source":     chunk.get("source"),
        }
    except Exception as e:
        print(f"  [WARN] GT generation failed: {e}")
        return None
 
 
def _sample_evenly(chunks: list, n: int, seed: int = config.RANDOM_SEED) -> list:
    """Sample n chunks spread evenly across all categories."""
    by_cat = defaultdict(list)
    for c in chunks:
        by_cat[c.get("category", "unknown")].append(c)
 
    rng     = random.Random(seed)
    sampled = []
    cats    = list(by_cat.keys())
    per_cat = max(1, n // len(cats))
 
    for cat in cats:
        pool    = by_cat[cat]
        sampled.extend(rng.sample(pool, min(per_cat, len(pool))))
 
    # Top up to exactly n if needed
    remaining = [c for c in chunks if c not in sampled]
    rng.shuffle(remaining)
    sampled.extend(remaining[: max(0, n - len(sampled))])
    return sampled[:n]
 
 
def generate_gt_pairs(
    oai: OpenAI,
    qdrant: QdrantClient,
    collection_name: str,
    n: int = config.GT_PAIRS,
    save_path: str = config.GT_PAIRS_PATH,
) -> list:
    """
    Generate n ground-truth QA pairs by sampling chunks evenly from Qdrant.
    Saves to save_path (JSON) and returns the list.
 
    Only needs to be run once — reuse gt_pairs.json across all eval configurations.
 
    Usage:
        from eval.generate_gt import generate_gt_pairs
        gt_pairs = generate_gt_pairs(oai, qdrant, collection_name="hr_rag_recursive")
    """
    print(f"Loading chunks from '{collection_name}'...")
    all_points, _ = qdrant.scroll(
        collection_name=collection_name,
        limit=10_000,
        with_payload=True,
    )
    data   = [p.payload for p in all_points]
    chunks = _sample_evenly(data, n)
    print(f"Sampled {len(chunks)} chunks across {len(set(c.get('category') for c in chunks))} categories.")
 
    gt_pairs = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i:02d}/{n}] {chunk.get('category')} ...", end=" ", flush=True)
        pair = _call_openai(chunk, oai)
        if pair:
            gt_pairs.append(pair)
            print("✓")
        else:
            print("✗")
        time.sleep(0.4)
 
    print(f"\nGenerated {len(gt_pairs)} GT pairs.")
 
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(gt_pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved to {save_path}")
 
    return gt_pairs
 
 
def load_gt_pairs(path: str = config.GT_PAIRS_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)