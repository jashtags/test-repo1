# PDF Intelligence Platform

A local web app that ingests complex technical PDFs, lets you query them with natural language, and generates executive-level strategic business briefs — with built-in guardrails that catch jargon and flag contradictions across documents.

Runs entirely on your machine. No cloud LLM costs. Uses Ollama to run a local AI model.

---

## What You Built

You built a **RAG (Retrieval-Augmented Generation) application** — a category of AI tool that:
1. Reads and understands your documents
2. Stores their meaning as mathematical vectors in a local database
3. When you ask a question, finds the most relevant parts of your documents
4. Feeds those parts to an AI model to generate a grounded, cited answer

This is the same architecture used by enterprise tools like Notion AI, Perplexity, and Microsoft Copilot — but running 100% locally and for free.

---

## Technology Stack — What Was Used and Why

### Streamlit
**What:** Python library that turns Python scripts into interactive web apps.  
**Why:** Lets you build a full UI (file uploaders, tabs, streaming text, progress bars) in pure Python — no HTML, CSS, or JavaScript needed. Ideal for data and AI tools.

### LlamaParse
**What:** Cloud-based PDF parser from LlamaIndex.  
**Why:** Standard PDF readers (like PyPDF) extract raw text and lose all formatting. LlamaParse preserves:
- Tables rendered as proper markdown tables
- Mathematical equations rendered in LaTeX notation
- Section headers, figure captions, footnotes

This matters for ML research papers and market reports where tables and formulas carry the key information.

### LangChain
**What:** Python framework for building LLM-powered applications.  
**Why:** Acts as the glue between all the components. It provides:
- LCEL (LangChain Expression Language) — a pipe `|` syntax to chain steps together
- Retriever abstractions — consistent interface to query any vector database
- Prompt templates — structured prompts sent to the LLM
- Output parsers — convert LLM output to clean strings

Without LangChain you would manually wire together the retrieval, prompt formatting, and LLM calls.

### ChromaDB
**What:** Local vector database that persists to disk.  
**Why:** After text is converted to vectors (numbers representing meaning), it needs to be stored somewhere searchable. ChromaDB:
- Runs entirely on your machine, no server or cloud account needed
- Persists to `./chroma_db` folder — survives app restarts
- Performs fast similarity search across thousands of chunks in milliseconds

### HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
**What:** A local ML model that converts text into 384-dimensional vectors.  
**Why:** Every chunk of text and every question you type gets converted to a vector (list of 384 numbers). Chunks with similar meaning end up with similar numbers — so searching by meaning is just finding the nearest numbers. This model:
- Runs locally, no API key
- Is small enough (80MB) to run on CPU
- Is fast and accurate for English semantic search

### Ollama + Llama 3.2 (3B)
**What:** Ollama is a tool to run large language models locally. Llama 3.2 3B is Meta's open-source model.  
**Why:** Originally built for Claude (Anthropic API) but switched to Ollama to eliminate costs. Ollama:
- Downloads and manages model weights on your Mac
- Exposes a local API that LangChain talks to
- No internet needed once the model is downloaded
- Completely free, no usage limits

The 3B model is lightweight enough to run on a Mac without a GPU.

### Guardrails (custom-built, no LLM)
Two programmatic safety layers:

**Jargon Rewriter** — regex-based find-and-replace that scans all AI outputs for aggressive market language before it reaches stakeholders:

| Aggressive | Strategic |
|-----------|-----------|
| disrupt | innovate |
| dominate | lead the market |
| crush | outperform |
| aggressive | strategic |
| arms race | capability investment |
| capture the market | grow market share |
| wipe out | outcompete |

**Contradiction Detector** — scans across multiple ingested documents looking for chunks that talk about similar topics but report different numbers (e.g. two market reports cite different market sizes). Flags these with severity ratings.

---

## Full Process Flow — From PDF Upload to Answer

```
INGEST PHASE
────────────
1. You upload a PDF in the browser

2. LlamaParse sends it to LlamaCloud API
   → Returns markdown with tables and LaTeX equations preserved
   → Falls back to PyPDF if no API key

3. Text split into overlapping chunks
   → Each chunk: ~1000 characters with 200 character overlap
   → Overlap ensures sentences are not cut off at chunk boundaries

4. HuggingFace model converts each chunk to a vector
   → 384 numbers representing the semantic meaning of that chunk

5. Vectors + original text stored in ChromaDB on disk
   → Saved to ./chroma_db — persists between sessions


QUERY PHASE
───────────
6. You type a question in the Query tab

7. Same HuggingFace model converts your question to a vector

8. ChromaDB finds the top-k chunks with vectors closest to
   your question vector (cosine similarity search)

9. Those chunks are formatted with their source filename and page number

10. Prompt assembled and sent to Llama 3.2 via Ollama:
    ┌──────────────────────────────────────────────────────┐
    │ System: You are a precise document analyst...         │
    │ Context: [chunk1] [chunk2] [chunk3] [chunk4]          │
    │ Question: your question here                          │
    └──────────────────────────────────────────────────────┘

11. Llama 3.2 generates an answer running locally on your Mac

12. Answer streams back token by token to the Streamlit UI

13. Jargon Rewriter scans the answer automatically
    → Replaces aggressive language with strategic alternatives

14. Source chunks shown below the answer so you can verify


EXECUTIVE BRIEF PHASE
─────────────────────
Same retrieval as Query but with a different prompt that instructs
the model to output a structured executive brief:
  → Executive Summary
  → Market and Competitive Landscape
  → Risk / Opportunity Matrix (table)
  → Recommended Strategic Initiatives
  → Key Performance Indicators
  → Quarterly Implementation Roadmap
  → Strategic Recommendation
```

---

## File Structure

```
test-repo1/
│
├── app.py            ← Main Streamlit UI. Defines all 4 tabs and wires
│                       everything together. Run this file to start the app.
│
├── parser.py         ← PDF ingestion. Tries LlamaParse first (cloud, preserves
│                       tables and equations), falls back to PyPDF (free, basic).
│
├── vectorstore.py    ← ChromaDB setup. Splits documents into chunks,
│                       embeds them with HuggingFace, stores to disk.
│
├── chains.py         ← LangChain LCEL chains.
│                       build_qa_chain()               — RAG question answering
│                       build_executive_summary_chain() — structured briefs
│
├── guardrails.py     ← Two guardrails (no LLM used):
│                       rewrite_jargon()        — regex language normalization
│                       detect_contradictions() — cross-document conflict detection
│
├── requirements.txt  ← All Python dependencies
└── .env.example      ← Template for API keys (copy to .env and fill in)
```

---

## Setup and Run

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed with a model pulled:
  ```bash
  ollama pull llama3.2
  ```

### Install and run

```bash
# Clone the repo
git clone https://github.com/jashtags/test-repo1.git
cd test-repo1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys (LlamaCloud is optional — falls back to PyPDF without it)
cp .env.example .env
# Edit .env and add your LLAMA_CLOUD_API_KEY if you have one

# Run
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## The 4 Tabs

| Tab | What it does |
|-----|-------------|
| **Ingest Documents** | Upload PDFs. LlamaParse extracts text preserving tables and equations. Chunks are embedded and stored in ChromaDB. |
| **Query** | Ask natural language questions. Retrieves the most relevant chunks and feeds them to the local LLM for a grounded, cited answer. |
| **Executive Brief** | Enter a strategic topic. Generates a full CSO-level brief with Risk/Opportunity Matrix, KPIs, and quarterly roadmap — grounded in your documents. |
| **Guardrails** | (1) Paste text to strip aggressive jargon. (2) Run contradiction detection across all ingested documents to flag conflicting numerical claims. |

---

## Why This Architecture

The core design decision is **separating search from generation**:

- **Search** (ChromaDB + HuggingFace embeddings) finds the right information — fast, deterministic, free
- **Generation** (Ollama + Llama 3.2) writes the answer — creative, contextual, local

The LLM only ever sees the relevant chunks, not the entire document. This:
- Prevents hallucination — the model cannot fabricate facts not in the retrieved chunks
- Makes answers verifiable — you can see exactly which source chunks were used
- Keeps it fast — fewer tokens processed per query

The guardrails are intentionally **not LLM-based**. Using regex and keyword analysis means they are:
- Deterministic — same input always gives the same output
- Instant — no model inference, no latency
- Auditable — you can read exactly which rules fired and why
- Free — zero token cost
