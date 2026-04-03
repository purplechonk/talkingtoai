# Getting Started — Local Setup

---

## Which situation are you in?

```
First time on this machine?
  → Follow SETUP (Steps 1–5) then PATH A

Already done setup, chunks already in Qdrant?
  → Skip to PATH B (run eval or explore)

Already done setup, want to add new sources?
  → Skip to PATH C (re-scrape and re-index)
```

---

## One-time Setup (do this once per machine)

### Step 1 — Clone and enter the repo

```bash
git clone https://github.com/YOUR_USERNAME/talkingtoai.git
cd talkingtoai
```

---

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

> Every time you open a new terminal, re-run the `activate` line before anything else.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> First install takes ~5 minutes — pulls PyTorch, LangChain, RAGAS, fastembed, etc.

---

### Step 4 — Set up API keys

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in:

```
OPENAI_API_KEY=sk-...          # required for embeddings, LLM tagging, and eval

QDRANT_URL=https://<cluster-id>.<region>.gcp.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>
```

> `.env` is git-ignored and never committed. Get your Qdrant URL + API key from [cloud.qdrant.io](https://cloud.qdrant.io).

---

### Step 5 — Launch Jupyter

```bash
jupyter notebook notebooks/
```

Your browser should open automatically. You're now ready to pick a path below.

---

---

## PATH A — First time: scrape → chunk → embed → eval

**Do this the very first time. Takes ~2–4 hours total.**

### A1. Run `01_scraping.ipynb` (once)

Scrapes all sources defined in `data/sources.json` and builds your corpus.

1. Open `notebooks/01_scraping.ipynb`
2. Run all cells top-to-bottom
3. Wait for it to finish (20–40 min, network dependent)

Expected output at the end:
```
data/
  downloads/           ← raw PDF files
  scraped_corpus.json  ← cleaned documents  ✅
```

> If a URL fails, the scraper logs it and continues — check the failure list printed in the last cell.

---

### A2. Run `02_build_index.ipynb` (once per strategy)

Chunks the corpus, tags metadata, embeds, and uploads to Qdrant Cloud.

1. Open `notebooks/02_build_index.ipynb`
2. Run Cells 1–3 (environment + clients + load corpus)
3. In **Cell 4**, set the strategy you want to build:
   ```python
   STRATEGY = 'recursive'   # start here
   ```
4. Run Cell 4 and wait (~10–30 min per strategy)
5. Run Cell 5 to verify the collection point count in Qdrant

**To build all 5 strategies**, change `STRATEGY` in Cell 4 and re-run Cell 4 only (no need to re-run Cells 1–3):

```python
# Run Cell 4 five times, once for each:
STRATEGY = 'recursive'
STRATEGY = 'semantic'
STRATEGY = 'sentence_window'
STRATEGY = 'parent_child'
STRATEGY = 'proposition'
```

> You need **at least `recursive`** before running eval. The others are optional but needed for a full comparison.

One Qdrant collection is created per strategy:

| Collection name | Strategy |
|---|---|
| `hr_rag_recursive` | LangChain RecursiveCharacterTextSplitter |
| `hr_rag_semantic` | SemanticChunker (topic-boundary splits) |
| `hr_rag_sentence_window` | Small chunks + surrounding sentence context |
| `hr_rag_parent_child` | Small retrieval chunks + large parent context |
| `hr_rag_proposition` | LLM-extracted atomic fact chunks |

Expected outputs after indexing:
```
data/
  chunks/recursive.json       ← saved chunk file per strategy
  embeddings/recursive.json   ← embedding cache (prevents re-embedding on reruns)
```

---

### A3. Run `03_run_eval.ipynb`

Now go to PATH B below — it's the same from here.

---

---

## PATH B — Already indexed: run eval or explore queries

**Use this when Qdrant already has your collections loaded.**

### Option B1 — Run the full evaluation (`03_run_eval.ipynb`)

Benchmarks every `(chunking strategy × retrieval strategy)` combination using RAGAS metrics.

1. Open `notebooks/03_run_eval.ipynb`
2. Run Cells 1–3 (environment + clients)
3. **Cell 4** generates 50 ground-truth QA pairs — only calls OpenAI the first time; reuses `data/gt_pairs.json` on reruns
4. **Cell 5** runs the eval grid — change these to limit scope and cost:
   ```python
   chunking_strategies  = ['recursive'],          # or all 5
   retrieval_strategies = ['dense', 'hybrid'],    # or all 4
   ```
5. Results saved to `results/eval_results.csv`

**Run time:** ~1–2 hours for full grid. ~15–20 min for a 1 chunking × 2 retrieval subset.

> **Tip:** Run a subset first (e.g. `recursive` + `dense`) to confirm everything works end-to-end before launching the full grid.

---

### Option B2 — Explore queries interactively (`04_explore.ipynb`)

Test individual questions and compare configs side by side without any eval overhead.

1. Open `notebooks/04_explore.ipynb`
2. Run Cells 1–2 (environment + clients)
3. In **Cell 3**, change the query and config:
   ```python
   result = rag(
       query    = "What are an employee's rights when requesting flexible work?",
       chunking  = 'recursive',        # recursive | semantic | sentence_window | parent_child | proposition
       strategy  = 'hybrid',           # dense | hybrid | multi_query | compression
       oai=oai, qdrant=qdrant, bm25_model=bm25_model,
       openai_api_key=OPENAI_API_KEY,
   )
   print(result['answer'])
   ```
4. **Cell 4** runs two configs side by side so you can directly compare answers

---

### Option B3 — Run eval from the command line (no Jupyter)

```bash
# Generate ground-truth pairs only
python eval/generate_gt.py

# Run full evaluation grid
python eval/evaluate.py

# Limit scope
python eval/evaluate.py --chunk recursive --ret dense,hybrid
```

---

---

## PATH C — Adding new sources and re-indexing

**Use this when you've added new PDFs or URLs to `data/sources.json`.**

1. Add your new sources to `data/sources.json`
2. Re-run `01_scraping.ipynb` — it will scrape everything and overwrite `scraped_corpus.json`
3. Re-run Cell 4 of `02_build_index.ipynb` for each strategy you want to update

**Will this create duplicates in Qdrant?** No. Chunk IDs are deterministic (hash-based), so:
- Unchanged content → same ID → **replaced in-place**
- New content → new ID → **inserted**
- Removed content → old ID **lingers** (orphaned)

If you want a completely clean rebuild (e.g. after removing sources), set `recreate=True` in Cell 4:
```python
build_index(
    ...,
    recreate=True,   # wipes the collection first, then rebuilds from scratch
)
```

---

---

## Estimated OpenAI API costs

| Step | Model | Approximate cost |
|---|---|---|
| NB02 — metadata tagging | gpt-4o-mini | ~$0.50–$2 per strategy |
| NB02 — embedding | text-embedding-3-small | ~$0.10–$0.30 per strategy |
| NB02 — proposition chunking | gpt-4o-mini | ~$1–$3 (LLM per chunk) |
| NB03 — GT generation | gpt-4o-mini | ~$0.20 for 50 pairs |
| NB03 — RAGAS eval | gpt-4o-mini | ~$1–$3 per config |

Full grid (5 chunking × 4 retrieval = 20 configs) costs roughly **$20–$40** total.
**Recommended first run:** `recursive` + `dense` + `hybrid` only (~$3–$5).

---

## Project structure

```
talkingtoai/
├── .env                    ← your secrets (git-ignored, create from .env.example)
├── .env.example            ← template
├── config.py               ← all settings in one place
├── requirements.txt
│
├── data/
│   ├── sources.json        ← URL/PDF manifest (committed to git)
│   ├── scraped_corpus.json ← scraped text (git-ignored, created by NB01)
│   ├── chunks/             ← chunk files per strategy (git-ignored)
│   ├── embeddings/         ← embedding cache per strategy (git-ignored)
│   ├── downloads/          ← raw PDFs (git-ignored)
│   └── gt_pairs.json       ← ground-truth QA pairs (git-ignored)
│
├── pipeline/
│   ├── chunkers.py         ← 5 chunking strategies
│   ├── embedder.py         ← metadata tagging + embedding + Qdrant upsert
│   ├── retriever.py        ← 4 retrieval strategies + cross-encoder reranking
│   └── rag.py              ← full RAG pipeline (query → retrieve → generate)
│
├── eval/
│   ├── generate_gt.py      ← ground-truth QA generation
│   └── evaluate.py         ← RAGAS grid evaluation
│
├── notebooks/
│   ├── 01_scraping.ipynb   ← scrape sources → scraped_corpus.json
│   ├── 02_build_index.ipynb← chunk + embed + upsert to Qdrant
│   ├── 03_run_eval.ipynb   ← RAGAS benchmark across all configs
│   └── 04_explore.ipynb    ← interactive query testing
│
└── results/
    └── eval_results.csv    ← comparison table (created by NB03)
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'config'`**
Make sure you launched Jupyter from the project root (`jupyter notebook notebooks/`), not from inside `notebooks/`. The setup cell resolves the path automatically.

**`AuthenticationError` from OpenAI**
Check that `.env` exists and `OPENAI_API_KEY=sk-...` is set. The key must start with `sk-`.

**`Unauthorized` or `Connection refused` for Qdrant**
Check that `QDRANT_URL` and `QDRANT_API_KEY` in `.env` match your cluster at [cloud.qdrant.io](https://cloud.qdrant.io).

**Scraping notebook skips a PDF**
Some government PDFs block automated downloads. The scraper logs failures and continues. You can manually download the PDF, place it in `data/downloads/`, and it will be picked up on the next run.

**Semantic or proposition chunking is very slow**
Both make OpenAI API calls per chunk. Start with `recursive` to validate the full pipeline end-to-end, then add the slower strategies once you're confident everything works.

**Embedding cache is stale after changing chunk size**
Delete `data/embeddings/<strategy>.json` — it will be rebuilt fresh on the next run of NB02.
