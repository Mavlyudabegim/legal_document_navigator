# ⚖️ Legal Document Navigator

Upload a contract (PDF), ask questions in plain language, and get short answers that cite the
exact clause. The PDF opens next to the chat and the sentences the answer relied on are
highlighted, so every answer can be checked in seconds.

> This tool explains documents. It is not legal advice.

**Live demo:** _add your Streamlit Cloud link here_

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment the line below -->
<!-- ![screenshot](docs/screenshot.png) -->

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
- **Cloud storage.** PDFs, vectors and chats are stored in Supabase. Without Supabase it falls
  back to local Chroma.
- **Evaluated on a public benchmark** (ContractNLI), not just on hand-picked examples.

## How it works

```
ingestion (once per PDF)
  PDF ──► parser ──► clause chunker ──► embeddings (bge-small) ──► Supabase / Chroma
          (PyMuPDF)   (sections, ≤350 tokens)                       chunks + vectors

question
  question ──► hybrid search (embeddings + BM25, RRF) ──► top-5 clauses
           ──► LLM (gpt-oss-20b on Groq) with a strict "cite or say not found" prompt
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

Measured on **ContractNLI** (Koreeda & Manning, 2021): real NDAs where lawyers labelled 17
statements per contract as _entailed_, _contradicted_ or _not mentioned_, and marked the
evidence sentences. Each statement is asked as a question, and the answer must start with
Yes / No / "The document doesn't say this".

**Answer accuracy** (dev split, 60 questions; model gpt-oss-20b):

| Setup                                                   | Accuracy                     |
| ------------------------------------------------------- | ---------------------------- |
| Baseline RAG                                            | 65.0%                        |
| + stricter prompt (explicit evidence, check exceptions) | 73.3%                        |
| + questions phrased like the labelling guidelines       | **78.3%** (95% CI 66–87%)    |
| Whole contract in the prompt instead of search          | 71.7% (worse)                |
| High reasoning effort                                   | 75.0% (and about 10× slower) |

| Metric           | This project | ContractNLI paper (fine-tuned BERT-large) |
| ---------------- | ------------ | ----------------------------------------- |
| Accuracy         | 0.783        | 0.875                                     |
| F1 Contradiction | 0.462        | 0.357                                     |
| F1 Entailment    | 0.852        | 0.834                                     |

The paper's model was trained on this task, while this project uses an off-the-shelf LLM
with no training. The paper's numbers are on the full test set and these are on a 60-question
dev sample, so the comparison is only indicative.

**Retrieval** (202 questions with lawyer-marked evidence):

| Embeddings, chunk size       | Hit@1 | Hit@3 | Hit@5 | Evidence recall |
| ---------------------------- | ----- | ----- | ----- | --------------- |
| bge-small, 350 tokens (used) | 55.9% | 82.2% | 95.5% | 93.4%           |
| bge-large, 500 tokens        | 62.4% | 92.1% | 96.5% | 94.7%           |

bge-large found the evidence more often but gave the **same** answer accuracy (78.3%), so the
smaller and faster model is used. Most remaining errors come from the answer step, not the search.

Reproduce:

```bash
# 1. download ContractNLI from https://stanfordnlp.github.io/contract-nli/ into data/contract-nli/
python -m eval.contractnli prepare --split dev --n-docs 20
python -m eval.contractnli run --configs hybrid --llm --max-questions 60
python -m eval.contractnli mistakes        # where each error comes from: search or answer
```

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

Terminal tools:

```bash
python -m app.ingestion.ingest data/legal_document_sample.pdf    # index a PDF (local Chroma)
python ask.py                                                    # chat in the terminal
```

## Cloud storage (Supabase)

1. Create a free project at [supabase.com](https://supabase.com).
2. In **SQL Editor**, run [`supabase/setup.sql`](supabase/setup.sql). It creates the
   `documents`, `chunks` and `chats` tables (with pgvector) and a private `uploads` bucket.
3. Copy the **Project URL** and the **secret key** (Project Settings → API Keys) into `.env`
   as `SUPABASE_URL` and `SUPABASE_KEY`.

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) choose **Create app**, then this
   repository, branch `main`, and file `streamlit_app.py`. Under Advanced settings pick Python 3.12.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "..."
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
  generation/           prompt, LLM client (Groq / OpenAI / Ollama), answer + citation check
  storage/              Supabase (cloud) and local-folder storage with the same interface
  ui/                   page helpers: PDF rendering + highlighting, chat (de)serialisation, CSS
eval/                   ContractNLI benchmark + small hand-written test set
supabase/setup.sql      database schema for the cloud version
```

## Tech stack

Python · Streamlit · PyMuPDF · sentence-transformers (BAAI/bge-small-en-v1.5) · rank-bm25 ·
Groq (gpt-oss-20b, OpenAI-compatible API) · Supabase (Postgres + pgvector + Storage) · ChromaDB

## Limitations

- English contracts with a text layer only. Scanned PDFs would need OCR.
- About 78% agreement with lawyers on ContractNLI. Always check the cited clause.
- With the free Groq tier, many questions in a row can hit rate limits.

## Acknowledgements

ContractNLI dataset by Yuta Koreeda and Christopher D. Manning, _"ContractNLI: A Dataset for
Document-level Natural Language Inference for Contracts"_ (Findings of EMNLP 2021), licensed
CC BY 4.0.
