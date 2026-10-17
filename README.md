# ⚖️ Legal Document Navigator

Upload some legal document or contract (PDF), ask questions in plain language, and get short answers in plain language that cite the
exact clause. The PDF opens next to the chat and the sentences the answer relied on are
highlighted, so every answer can be checked in seconds.

> This tool explains documents. It is not legal advice.

**Live demo:** [Legal Document Navigator](https://legaldocumentnavigator.streamlit.app/)


## Results at a glance

Evaluated on **ContractNLI**, real NDAs labelled by lawyers. The pipeline is the same in every
row; only the language model changes.

|                                                    | gpt-oss-20b                  | **gpt-oss-120b**                   | Fine-tuned BERT (paper)   |
| -------------------------------------------------- | ---------------------------- | ---------------------------------- | ------------------------- |
| Accuracy                                           | 75.3% (300 q, 95% CI 70–80%) | **85.0%** (100 q, 95% CI 77–91%)   | 87.5%                     |
| Same 100 questions                                 | 80 correct                   | **85 correct** (6 fixed, 1 broken) | —                         |
| Answers citing the clause that proves them         | 93.5%                        | **98.4%**                          | —                         |
| Strict accuracy (right label _and_ right citation) | 71.0%                        | **83.0%**                          | —                         |
| Seconds per question                               | ~28 (local, Ollama)          | **~15** (Groq)                     | —                         |
| Training on the task                               | none                         | none                               | fine-tuned on ContractNLI |

The 120b result is on 100 of the 300 fixed dev questions; the remaining 200 are in progress.
The paper reports the full test split. Details and caveats are under [Evaluation](#evaluation).

## Features

- **Layout-aware PDF parsing.** Headings are found by font size and bold text, headers and
  footers are removed, and clause numbering (1., 2.3, (a)) is kept.
- **Clause-based chunking.** Chunks follow the contract's sections instead of fixed-size
  windows, and each chunk carries its heading path, e.g. "MUTUAL NDA > 4. Term and Termination".
- **Hybrid search.** Embeddings (meaning) and BM25 (exact legal terms) are merged with
  Reciprocal Rank Fusion.
- **Grounded answers with verified citations.** The model may only use the given clauses.
  Citations to clauses it was never given are removed, and if nothing relevant is found it
  says so instead of guessing.
- **Source viewer.** The cited page opens with the 1–2 most relevant sentences highlighted.
- **Private by default.** Each browser gets its own document list (no login needed).
- **Cloud storage.** PDFs, vectors and chats are stored in Supabase (Postgres + pgvector +
  Storage). Without Supabase it falls back to local Chroma.
- **Evaluated on a public benchmark** with fixed samples, significance tests, citation checks
  and error analysis.

## How it works

<img width="1418" height="961" alt="image" src="https://github.com/user-attachments/assets/22bd2692-1ac9-4b2e-bf2e-fc635fc35a6b" />


```
ingestion (once per PDF)
  PDF ──► parser ──► clause chunker ──► embeddings (bge-small) ──► Supabase / Chroma
          (PyMuPDF)   (sections, ≤350 tokens)                       chunks + vectors

question
  question ──► hybrid search (embeddings + BM25, RRF) ──► top-5 clauses
           ──► LLM (gpt-oss on Groq) with a strict "cite or say not found" prompt
           ──► citation check ──► answer with [1] [2] ──► highlighted PDF page
```

| Step                                   | Code                                     |
| -------------------------------------- | ---------------------------------------- |
| PDF → structured elements              | `app/ingestion/parser.py`                |
| Elements → clause chunks               | `app/ingestion/chunker.py`               |
| Embeddings                             | `app/ingestion/embedder.py`              |
| Hybrid retrieval (+ optional reranker) | `app/retrieval/`                         |
| Prompt, LLM call, citation check       | `app/generation/`                        |
| Storage: Supabase or local             | `app/storage/`, `app/ingestion/store.py` |
| Web interface                          | `streamlit_app.py`, `app/ui/`            |
| Evaluation                             | `eval/`                                  |

## Evaluation

### Benchmark and method

**ContractNLI** (Koreeda & Manning, 2021): for each NDA, lawyers labelled 17 statements as
_Entailment_, _Contradiction_ or _NotMentioned_ and marked the evidence sentences. Each
statement is asked as a question ("Does the agreement say …?"). The answer must start with
Yes / No / "The document doesn't say this", which is graded automatically. An LLM judge is
used only for the few answers that start with "Partly".

- **Fixed sample.** 300 questions from all 61 NDAs of the dev split, saved in
  `eval/samples/`, so every configuration answers exactly the same questions. It contains
  **all 95 Contradiction questions** (only 9% of ContractNLI, too few in a random sample) plus
  Entailment and NotMentioned in their natural ratio. Headline figures are also re-weighted
  to the real label mix.
- **Paired comparisons.** Configurations are compared question by question (McNemar test),
  with 95% intervals that resample whole contracts, because questions about the same
  contract are related.
- **Citation checks.** A citation counts only if the cited clause contains the evidence
  sentence the lawyers marked.

### Main results

| Metric                                                              | gpt-oss-20b (300 q)   | gpt-oss-120b (100 q)  |
| ------------------------------------------------------------------- | --------------------- | --------------------- |
| Accuracy (95% CI)                                                   | 75.3% (70–80%)        | **85.0%** (77–91%)    |
| Accuracy, re-weighted to the real label mix                         | 76.2%                 | **85.8%**             |
| Macro-F1                                                            | 0.752                 | **0.848**             |
| F1 Entailment / Contradiction / NotMentioned                        | 0.777 / 0.762 / 0.718 | 0.864 / 0.842 / 0.839 |
| F1 at the real label mix: Entailment / Contradiction / NotMentioned | 0.811 / 0.570 / 0.761 | 0.893 / 0.694 / 0.858 |
| Empty answers                                                       | 5                     | 0                     |

**Same 100 questions, both models:** gpt-oss-20b answered 80 correctly, gpt-oss-120b 85. 120b
fixed 6 answers and broke 1. Every earlier experiment fixed about as many answers as it broke;
this is the first one-sided change. With 7 differing answers it is not yet statistically
significant (McNemar p ≈ 0.13), and these 100 questions were easier than average (20b: 80%
here vs 75.3% on all 300). The full 300-question run will settle it.

Compared with the paper (real label mix):

| Metric           | gpt-oss-120b (this project) | Span NLI BERT-large (paper, fine-tuned) |
| ---------------- | --------------------------- | --------------------------------------- |
| Accuracy         | 0.858                       | 0.875                                   |
| F1 Contradiction | 0.694                       | 0.357                                   |
| F1 Entailment    | 0.893                       | 0.834                                   |

The paper's model was trained on this task and evaluated on the full test split; this
project uses off-the-shelf models with no training, measured on the dev split.

### Citation quality

| Metric                                                              | gpt-oss-20b | gpt-oss-120b |
| ------------------------------------------------------------------- | ----------- | ------------ |
| Yes/no answers whose citation contains the evidence                 | 93.5%       | **98.4%**    |
| Citation precision (cited clauses that contain evidence)            | 87.3%       | 87.6%        |
| Citation recall (evidence sentences that were cited)                | 89.2%       | **95.2%**    |
| Right label but unsupported citation ("right answer, wrong reason") | 8.3%        | **3.4%**     |
| **Strict accuracy** (correct label _and_ supporting citation)       | 71.0%       | **83.0%**    |

### Retrieval

614 questions with evidence, 61 NDAs: Hit@1 59.0% · Hit@3 88.4% · Hit@5 96.1% · MRR 0.74 ·
evidence recall 94.0%. With gpt-oss-20b, 23 of 26 missed contradictions had the evidence in
the retrieved clauses, and with gpt-oss-120b, 4 of 5. **The bottleneck was reading, not
search**, which is why changing the model helped and changing the search did not.

### Error analysis

The same patterns cause most errors with both models:

| Pattern                              | Example                                                                                                               | Typical wrong answer                                                                 |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **"Only" statements**                | Q: "…includes _only_ technical information?" Contract: "including … financial, marketing and operational information" | "Doesn't say", often while the answer itself explains the contradiction              |
| **Prohibition with a carve-out**     | "shall not copy … _without the prior written consent_"                                                                | "Yes, allowed with consent"; the lawyers read it as a prohibition                    |
| **Alternatives in a definition**     | "clearly marked 'Confidential' _or_ clearly intended to be disclosed in confidence"                                   | "Yes, marking is required"; marking is only one option                               |
| **Prohibition inferred from a duty** | "must return or destroy all information"                                                                              | "No, copies can't be kept"; the lawyers say the contract is silent on keeping copies |

Evidence containing an **exception** ("except", "unless", "notwithstanding") lowers
Contradiction recall for both models (20b: 67% vs 76%; 120b: 67% vs 87%). Several remaining
"errors" follow the annotators' conventions rather than being clear misreadings. A
fine-tuned model learns those conventions from thousands of labels; prompting alone does not.

### Reproduce

```bash
# download ContractNLI from https://stanfordnlp.github.io/contract-nli/ into data/contract-nli/
python -m eval.contractnli prepare --split dev --n-docs 61
python -m eval.contractnli run --configs hybrid --llm --sample 300 --provider ollama --model gpt-oss:20b --tag base
python -m eval.contractnli run --configs hybrid --llm --sample 300 --limit 100 --provider groq --model openai/gpt-oss-120b --tag big
python -m eval.contractnli analyze                                  # per-label F1, citations, error analysis
python -m eval.contractnli compare --runs eval/results/A.json eval/results/B.json   # paired comparison
```

Runs are resumable (rate limits are fine; run the same command again). `--limit N` uses an
evenly spread subset of the fixed sample. The experiment protocol is in
[`eval/EXPERIMENTS.md`](eval/EXPERIMENTS.md).

## Run locally

```bash
git clone https://github.com/<your-username>/legal-document-navigator.git
cd legal-document-navigator
python -m venv .venv && source .venv/bin/activate      # or a conda environment
pip install -r requirements-dev.txt
cp .env.example .env                                   # then add your GROQ_API_KEY
streamlit run streamlit_app.py
```

Without `SUPABASE_URL` / `SUPABASE_KEY` everything is stored on your computer (`chroma_db/`,
`data/uploads/`, `data/chats/`). Those folders are in `.gitignore`.

Choose the model in `.env`:

```
LLM_MODEL=openai/gpt-oss-120b      # best results (Groq)
# LLM_PROVIDER=ollama              # free and offline: uses gpt-oss:20b on your computer
```

Terminal tools:

```bash
python -m app.ingestion.ingest data/legal_document_sample.pdf    # index a PDF (local Chroma)
python ask.py                                                    # chat in the terminal
```

## Cloud storage (Supabase)

1. Create a free project at [supabase.com](https://supabase.com).
2. In **SQL Editor**, run [`supabase/setup.sql`](supabase/setup.sql). It creates the
   `documents`, `chunks` and `chats` tables (with pgvector) and a private `uploads` bucket.
   For an older database, run [`supabase/add_owner.sql`](supabase/add_owner.sql) once.
3. Copy the **Project URL** and the **secret key** (Project Settings → API Keys) into `.env`
   as `SUPABASE_URL` and `SUPABASE_KEY`.

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) choose **Create app**, then this
   repository, branch `main`, and file `streamlit_app.py`. Under Advanced settings pick Python 3.12.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "..."
   LLM_MODEL = "openai/gpt-oss-120b"
   SUPABASE_URL = "https://your-project-id.supabase.co"
   SUPABASE_KEY = "..."
   ```
4. Deploy. The first start takes a few minutes, because it installs PyTorch (CPU) and
   downloads the embedding model.

## Project structure

```
streamlit_app.py        web interface
ask.py                  terminal chat
app/
  config.py             settings (models, top-k, ...) read from environment variables
  ingestion/            parser, chunker, embedder, local Chroma store, ingest()
  retrieval/            hybrid retriever, optional cross-encoder reranker
  generation/           prompt, LLM client (Groq / Ollama / OpenAI / Cerebras / OpenRouter), answer + citation check
  storage/              Supabase (cloud) and local-folder storage with the same interface
  ui/                   PDF rendering + highlighting, per-browser identity, chat (de)serialisation, CSS
eval/                   ContractNLI benchmark, metrics, reports, experiment protocol
supabase/               database schema for the cloud version
```

## Tech stack

Python · Streamlit · PyMuPDF · sentence-transformers (BAAI/bge-small-en-v1.5) · rank-bm25 ·
gpt-oss-120b / gpt-oss-20b (Groq, Ollama; OpenAI-compatible API) · Supabase (Postgres +
pgvector + Storage) · ChromaDB

## Limitations

- English contracts with a text layer only. Scanned PDFs would need OCR.
- Documents are private per browser, not per account: another browser or device starts empty.
- About 85% agreement with lawyers on ContractNLI (gpt-oss-120b, 100 dev questions). Always
  check the cited clause.
- The free Groq tier has daily limits; heavy use can hit them.

## Acknowledgements

ContractNLI dataset by Yuta Koreeda and Christopher D. Manning, _"ContractNLI: A Dataset for
Document-level Natural Language Inference for Contracts"_ (Findings of EMNLP 2021), licensed
CC BY 4.0.
