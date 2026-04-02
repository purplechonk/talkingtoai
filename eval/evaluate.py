# eval/evaluate.py
# RAGAS evaluation runner — extracted from rag.ipynb Cells 24-30.
# Runs every (chunking, retrieval) combination and saves a comparison table.
 
import itertools
import json
import time
import pandas as pd
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client.http.exceptions import ResponseHandlingException
 
from pipeline.rag import rag
from eval.generate_gt import FAILURE_CASES
 
import sys
sys.path.append("..")
import config
 
 
# ── RAGAS metric sets ─────────────────────────────────────────────────────────
 
# Failure cases: no reference answer available
FAILURE_METRICS = [faithfulness, answer_relevancy, context_precision]
 
# GT pairs: reference answer available, so context_recall can be computed
GT_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
 
 
# ── Safe RAG call with retry ──────────────────────────────────────────────────
 
def _safe_rag(retries=3, wait=5, **rag_kwargs) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            return rag(**rag_kwargs)
        except (ResponseHandlingException, Exception) as e:
            if attempt == retries:
                print(f"    [FAIL] {e}")
                return None
            print(f"    ⚠ Attempt {attempt} failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
 
 
# ── Single-configuration evaluation ──────────────────────────────────────────
 
def run_eval_one(
    chunking:       str,
    retrieval:      str,
    gt_pairs:       list,
    oai,
    qdrant,
    bm25_model,
    cross_encoder,
    openai_api_key: str,
    use_rerank:     bool = False,
    run_failure:    bool = True,
    run_gt:         bool = True,
) -> dict:
    """
    Run RAGAS evaluation for one (chunking, retrieval) configuration.
    Returns a dict of mean metric scores.
 
    Args:
        chunking      : one of config.CHUNKING_STRATEGIES
        retrieval     : one of config.RETRIEVAL_STRATEGIES
        gt_pairs      : list of GT dicts from generate_gt.py
        use_rerank    : apply cross-encoder reranking on top of retrieval
        run_failure   : evaluate on the 10 adversarial failure cases
        run_gt        : evaluate on the 50 GT pairs
 
    Returns:
        {
          "chunking": str, "retrieval": str, "rerank": bool,
          "faithfulness": float, "answer_relevancy": float,
          "context_precision": float, "context_recall": float,  # None if run_gt=False
          "n_failure": int, "n_gt": int,
        }
    """
    label = f"{chunking} / {retrieval}" + (" + rerank" if use_rerank else "")
    print(f"\n{'─'*60}")
    print(f"  Config: {label}")
    print(f"{'─'*60}")
 
    lc_llm = ChatOpenAI(model=config.LLM_MODEL, api_key=openai_api_key, temperature=0)
    lc_emb = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=openai_api_key)
 
    shared_rag_kwargs = dict(
        strategy=retrieval,
        chunking=chunking,
        oai=oai,
        qdrant=qdrant,
        bm25_model=bm25_model,
        cross_encoder=cross_encoder,
        openai_api_key=openai_api_key,
        top_k=config.TOP_K,
        use_rerank=use_rerank,
    )
 
    scores = {"chunking": chunking, "retrieval": retrieval, "rerank": use_rerank}
 
    # ── Failure case set ──────────────────────────────────────────────────────
    failure_scores = {}
    if run_failure:
        print(f"  Running {len(FAILURE_CASES)} failure cases...")
        rows = []
        for item in FAILURE_CASES:
            out = _safe_rag(query=item["query"], **shared_rag_kwargs, **item["filters"])
            if out:
                rows.append({
                    "user_input":         item["query"],
                    "response":           out["answer"],
                    "retrieved_contexts": out["contexts"],
                })
            time.sleep(0.3)
 
        if rows:
            res = evaluate(
                dataset=Dataset.from_list(rows),
                metrics=FAILURE_METRICS,
                llm=lc_llm,
                embeddings=lc_emb,
            )
            df = res.to_pandas()
            failure_scores = df[["faithfulness", "answer_relevancy", "context_precision"]].mean().to_dict()
            print(f"  Failure scores: {failure_scores}")
 
        scores["n_failure"] = len(rows)
 
    # ── GT pair set ───────────────────────────────────────────────────────────
    gt_scores = {}
    if run_gt and gt_pairs:
        print(f"  Running {len(gt_pairs)} GT pairs...")
        rows = []
        for item in gt_pairs:
            out = _safe_rag(query=item["user_input"], **shared_rag_kwargs)
            if out:
                rows.append({
                    "user_input":         item["user_input"],
                    "reference":          item["reference"],
                    "response":           out["answer"],
                    "retrieved_contexts": out["contexts"],
                })
            time.sleep(0.3)
 
        if rows:
            res = evaluate(
                dataset=Dataset.from_list(rows),
                metrics=GT_METRICS,
                llm=lc_llm,
                embeddings=lc_emb,
            )
            df = res.to_pandas()
            gt_scores = df[["faithfulness", "answer_relevancy",
                            "context_precision", "context_recall"]].mean().to_dict()
            print(f"  GT scores: {gt_scores}")
 
        scores["n_gt"] = len(rows)
 
    # Merge — GT scores overwrite failure scores where they overlap
    scores.update(failure_scores)
    scores.update(gt_scores)
 
    return scores
 
 
# ── Full grid evaluation ──────────────────────────────────────────────────────
 
def run_eval_grid(
    gt_pairs:       list,
    oai,
    qdrant,
    bm25_model,
    cross_encoder,
    openai_api_key: str,
    chunking_strategies:  list = None,
    retrieval_strategies: list = None,
    include_rerank:       bool = True,
    save_path:            str  = config.RESULTS_PATH,
) -> pd.DataFrame:
    """
    Run RAGAS evaluation across all (chunking × retrieval) combinations.
    Optionally also runs each combination with cross-encoder reranking.
    Saves results to CSV and returns a DataFrame.
 
    Usage:
        from eval.evaluate import run_eval_grid
        df = run_eval_grid(
            gt_pairs=gt_pairs,
            oai=oai, qdrant=qdrant, bm25_model=bm25_model,
            cross_encoder=cross_encoder, openai_api_key=OPENAI_API_KEY,
        )
        print(df.pivot_table(index="chunking", columns="retrieval", values="context_recall"))
    """
    chunking_strategies  = chunking_strategies  or config.CHUNKING_STRATEGIES
    retrieval_strategies = retrieval_strategies or config.RETRIEVAL_STRATEGIES
 
    all_results = []
    combos      = list(itertools.product(chunking_strategies, retrieval_strategies))
    total       = len(combos) * (2 if include_rerank else 1)
 
    print(f"Starting grid eval: {len(chunking_strategies)} chunkers × "
          f"{len(retrieval_strategies)} retrievers"
          + (" × 2 (with/without rerank)" if include_rerank else "")
          + f" = {total} configs\n")
 
    for i, (chunking, retrieval) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}]")
 
        # Without reranking
        result = run_eval_one(
            chunking=chunking, retrieval=retrieval,
            gt_pairs=gt_pairs, oai=oai, qdrant=qdrant,
            bm25_model=bm25_model, cross_encoder=cross_encoder,
            openai_api_key=openai_api_key, use_rerank=False,
        )
        all_results.append(result)
 
        # With reranking (only for strategies where it makes sense)
        if include_rerank and retrieval in ("dense", "hybrid"):
            result_rerank = run_eval_one(
                chunking=chunking, retrieval=retrieval,
                gt_pairs=gt_pairs, oai=oai, qdrant=qdrant,
                bm25_model=bm25_model, cross_encoder=cross_encoder,
                openai_api_key=openai_api_key, use_rerank=True,
            )
            all_results.append(result_rerank)
 
    df = pd.DataFrame(all_results)
 
    # Add a readable label column
    df["config"] = df.apply(
        lambda r: f"{r['chunking']} / {r['retrieval']}" + (" + rerank" if r["rerank"] else ""),
        axis=1,
    )
 
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")
 
    return df
 
 
def print_summary(df: pd.DataFrame):
    """Print a readable summary table after run_eval_grid."""
    metrics = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
               if c in df.columns]
    summary = df[["config"] + metrics].copy()
    for m in metrics:
        summary[m] = summary[m].round(4)
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(summary.sort_values("context_recall", ascending=False).to_string(index=False))
    print("="*70)