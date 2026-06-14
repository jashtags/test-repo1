# PDF Intelligence Platform

> Upload any technical PDF and understand it at **4 levels of increasing depth** — from zero-knowledge background to expert paper summary — powered by a fully local AI stack with an automatic judge that improves response quality.

---

## What Does It Do?

You upload a PDF (research paper, whitepaper, technical report) and the platform:

1. **Parses** the document preserving tables, equations, and structure
2. **Indexes** it into a local vector database for fast retrieval
3. **Generates** 4 progressive explanations — each building on the one before
4. **Judges** every response automatically and rewrites it if quality is low
5. **Answers** natural-language questions grounded in your documents

Everything runs **locally on your machine** — no data leaves your device, no cloud LLM costs.

---

## The 4 Learning Levels

Each level builds on the previous one. If you ingested a paper on transformer architecture, here is what each level gives you:

| Level | Name | Assumes You Know | What You Get |
|:-----:|------|:----------------:|--------------|
| 🌱 **1** | **Foundations** | Nothing at all | Complete beginner background: what it is, why it matters, main types, real-world use cases — explained with analogies, no jargon |
| ⚙️ **2** | **Concepts & Mechanisms** | Everything from Level 1 | HOW things work internally: architecture, data flow, design decisions, trade-offs at a conceptual level |
| 🔬 **3** | **Technical Deep Dive** | Levels 1 & 2 | Algorithmic specifics, mathematical intuition, benchmarks, limitations, comparisons with alternative approaches |
| 📄 **4** | **Expert Paper Summary** | Full domain expertise | Exact technical summary: problem statement, contributions, methodology, results with numbers, ablations, open questions |

### Example — "Attention Is All You Need" paper

| Level | Sample Output Style |
|:-----:|---------------------|
| 🌱 1 | "Imagine a library where instead of reading every book from cover to cover, you instantly jump to the most relevant pages. That's what attention does…" |
| ⚙️ 2 | "The encoder maps each token to a query, key, and value vector. Attention scores are computed as softmax(QKᵀ/√dₖ)V, allowing every position to attend to every other position…" |
| 🔬 3 | "Multi-head attention uses h=8 parallel heads with dₖ=dᵥ=64. Positional encodings use PE(pos,2i)=sin(pos/10000^(2i/d_model)), avoiding recurrence entirely…" |
| 📄 4 | "Vaswani et al. (2017) propose a sequence-to-sequence architecture relying entirely on attention, achieving 28.4 BLEU on WMT 2014 En→De vs 26.0 for the best prior ensemble…" |

---

## The Judge AI

After generating each level, a **judge LLM** (the same local Llama model, running at temperature=0 for consistency) evaluates the response:

```
┌─────────────────────────────────────────────────────┐
│                  Judge Evaluation                   │
│                                                     │
│  SCORE: 6/10          VERDICT: FAIL                 │
│  ─────────────────────────────────────────────────  │
│  Accuracy:   Factually correct but incomplete       │
│  Depth Fit:  Too technical for Level 1 audience     │
│  Clarity:    Jargon used without explanation        │
│  Missing:    Real-world use cases, analogies        │
│  Feedback:   Add at least 2 everyday analogies.     │
│              Define 'embedding' before using it.    │
│              Include 3 concrete use case examples.  │
│                                                     │
│  → Automatically generating improved response...    │
└─────────────────────────────────────────────────────┘
```

| Judge Criterion | What It Checks |
|----------------|----------------|
| **Accuracy** | Factual correctness vs. source document content |
| **Depth Fit** | Does the depth match the level's target audience? |
| **Clarity** | Is it accessible to the intended reader? |
| **Completeness** | What important content is missing? |
| **Score** | 1–10 integer; **≥ 7 = PASS**, < 7 triggers automatic improvement |

If the verdict is **FAIL**, the platform automatically generates an **improved response** with the judge's feedback incorporated — no action needed from you.

---

## Architecture

```
PDF Upload
    │
    ▼
┌────────────────────┐
│  LlamaParse        │  High-fidelity: tables, equations, captions
│  (or PyPDF)        │  Fallback: plain text extraction
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  ChromaDB          │  Local vector store, persists to disk
│  + HuggingFace     │  Embeddings: all-MiniLM-L6-v2 (384-dim)
│  Embeddings        │
└────────┬───────────┘
         │  RAG Retrieval (top-6 chunks)
         ▼
┌────────────────────┐
│  Llama 3.2 3B      │  Running locally via Ollama
│  Level 1-4 Chain   │  One chain per level, custom system prompt
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Judge LLM         │  Same Llama model, temperature=0
│  (Evaluator)       │  Scores: accuracy, depth fit, clarity
└────────┬───────────┘
         │
    score >= 7?
    ┌──────┴───────┐
   YES             NO
    │               │
    ▼               ▼
 PASS           Auto-Improve
             (re-run chain
              with feedback)
```

---

## Quick Start

> **Recommended for all users.** `justrun.py` handles everything automatically.

```bash
# 1. Clone the repo
git clone https://github.com/jashtags/test-repo1.git
cd test-repo1

# 2. Run the setup script
python justrun.py
```

`justrun.py` will walk you through an interactive setup in your terminal:

```
╔══════════════════════════════════════════════════════════════╗
║          PDF Intelligence Platform — Quick Start            ║
╚══════════════════════════════════════════════════════════════╝

── Checking prerequisites ──────────────────────────────────────
  ✓  Python 3.11.4
  ✓  pip found
  ✓  Ollama found
  ✓  Running from correct directory

── API Keys Setup ──────────────────────────────────────────────
  LlamaCloud API Key (Enter to skip): ****

── Installing Python dependencies ──────────────────────────────
  ✓  All dependencies installed

── Checking Ollama model ────────────────────────────────────────
  ✓  Model ready: hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF

── Launching app ────────────────────────────────────────────────
  Opening browser at http://localhost:8501
```

### What the script prompts you for

| Prompt | Required? | Where to get it |
|--------|:---------:|-----------------|
| LlamaCloud API Key | Optional | [cloud.llamaindex.ai](https://cloud.llamaindex.ai) — free tier available |

---

## Manual Setup

If you prefer to set things up yourself step by step:

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.9+ | [python.org](https://python.org) |
| Ollama | Latest | [ollama.ai](https://ollama.ai) |
| Git | Any | [git-scm.com](https://git-scm.com) |

### Steps

```bash
# 1. Clone
git clone https://github.com/jashtags/test-repo1.git
cd test-repo1

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and paste your LlamaCloud key (optional)

# 5. Pull the Llama model (first time only — approx 2 GB)
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest

# 6. Make sure Ollama is running
ollama serve    # skip if Ollama already runs as a background service

# 7. Launch
streamlit run app.py
```

---

## Using the App

### Tab 1 — Ingest Documents

1. Upload one or more PDFs using the file uploader
2. Choose **LlamaParse** (better quality, needs API key) or **PyPDF** (free fallback)
3. Click **Ingest Documents**
4. The vector store builds and persists to `./chroma_db` — it survives restarts

### Tab 2 — Query

Ask any question about your documents in plain English. Answers cite the source file and page number. Aggressive language is automatically normalized by the Jargon Guard.

### Tab 3 — Learning Levels

1. (Optional) Type a focus topic — e.g. `attention mechanism` or `training procedure`
2. Select a level tab: **Level 1**, **Level 2**, **Level 3**, or **Level 4**
3. Click **Generate Level N**
4. Watch the response stream in real time
5. The judge evaluates automatically — if quality is below 7/10, an improved version is generated

You can generate each level independently and regenerate any level at any time.

### Tab 4 — Guardrails

| Guardrail | What It Does |
|-----------|-------------|
| **Jargon Rewriter** | Replaces aggressive business language ("disrupt", "crush", "dominate") with preferred strategic terms |
| **Contradiction Detector** | Finds conflicting numerical claims across multiple ingested documents |

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **UI** | Streamlit | Fast prototyping, reactive state, no frontend code needed |
| **PDF Parsing** | LlamaParse / PyPDF | LlamaParse preserves tables and equations in markdown/LaTeX |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Fast, local, 384-dim, excellent for semantic search |
| **Vector Store** | ChromaDB | Persistent local store, no external service needed |
| **LLM** | Llama 3.2 3B via Ollama | Fully local inference, free after model download |
| **LLM Framework** | LangChain LCEL | Composable chains, streaming support |
| **Judge LLM** | Same Llama model (temp=0) | Deterministic evaluation using the same local model |
| **Orchestration** | Python 3.9+ | No special infrastructure |

---

## File Structure

```
test-repo1/
├── app.py            Main Streamlit application (4 tabs)
├── chains.py         LangChain chains: QA, 4 level chains, judge chain
├── guardrails.py     Jargon rewriter + contradiction detector
├── parser.py         PDF parsing (LlamaParse + PyPDF fallback)
├── vectorstore.py    ChromaDB setup and embedding management
├── justrun.py        One-command setup and launch script
├── requirements.txt  Python dependencies
├── .env.example      API key template
└── .gitignore        Excludes .env, chroma_db, __pycache__
```

---

## Frequently Asked Questions

**Q: Does this send my documents to any cloud service?**  
A: Only if you use LlamaParse for PDF parsing. The AI inference (Llama 3.2) runs entirely locally via Ollama. If you use PyPDF as the parser, nothing leaves your machine.

**Q: How long does generation take?**  
A: Depends on your hardware. On an Apple M1/M2 Mac, each level takes roughly 15–45 seconds. The judge adds about 10–15 seconds. If the judge triggers an improvement pass, add another 30 seconds.

**Q: Can I use a different Ollama model?**  
A: Yes — change `OLLAMA_MODEL` at the top of `chains.py`. Any model available via `ollama list` will work, including larger models like Llama 3.1 8B for better quality.

**Q: Do I need a LlamaCloud API key?**  
A: No. If you skip it, the app uses PyPDF to extract text. You lose table and equation formatting but every other feature works identically.

**Q: How does the judge decide to improve a response?**  
A: It scores the response 1–10 across accuracy, depth fit, clarity, and completeness. Any score below 7 triggers one automatic improvement pass with the judge's specific feedback incorporated into the system prompt.

**Q: Can I generate all 4 levels at once?**  
A: Each level is generated on-demand by clicking its "Generate" button. This lets you stop after Level 2 if that's all you need, saving time.

---

## License

MIT — use freely, attribution appreciated.
