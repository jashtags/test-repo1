import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PDF Intelligence Platform",
    page_icon="📚",
    layout="wide",
)

# ─── Session state ────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []  # filenames
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []  # LangChain Document objects (post-split)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    anthropic_key = ""  # Not needed — using local Ollama
    st.success("🦙 Using Ollama (local, free)")
    st.caption("Model: Llama-3.2-3B-Instruct")
    llama_key = st.text_input(
        "LlamaCloud API Key",
        value=os.getenv("LLAMA_CLOUD_API_KEY", ""),
        type="password",
        help="Required for LlamaParse (table + equation preservation). Falls back to PyPDF if blank.",
    )

    if st.session_state.ingested_docs:
        st.divider()
        st.subheader("Ingested Documents")
        for name in st.session_state.ingested_docs:
            st.write(f"• {name}")

    # Try loading existing vectorstore on first run
    if st.session_state.vectorstore is None:
        try:
            from vectorstore import load_vectorstore
            vs = load_vectorstore()
            if vs:
                st.session_state.vectorstore = vs
                st.info("Loaded existing vector store.")
        except Exception:
            pass

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("📚 PDF Intelligence Platform")
st.caption("LlamaParse · ChromaDB · LangChain · Llama 3.2 via Ollama")

tab_ingest, tab_query, tab_brief, tab_guard = st.tabs(
    ["📥 Ingest Documents", "🔍 Query", "📊 Executive Brief", "🛡️ Guardrails"]
)

# ─── Tab 1: Ingest ───────────────────────────────────────────────────────────
with tab_ingest:
    st.header("Document Ingestion")
    st.write(
        "Upload complex technical PDFs — ML papers, market reports, whitepapers. "
        "LlamaParse preserves tables and mathematical equations in markdown/LaTeX."
    )

    col_upload, col_opts = st.columns([2, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
    with col_opts:
        parser_choice = st.radio(
            "Parser",
            ["LlamaParse (recommended)", "PyPDF (fallback)"],
            help="LlamaParse requires a LlamaCloud API key.",
        )
        use_llamaparse = parser_choice.startswith("LlamaParse")

    if uploaded_files and st.button("Ingest Documents", type="primary"):
        from parser import parse_with_llamaparse, parse_with_fallback
        from vectorstore import ingest_documents

        new_docs = []
        progress_bar = st.progress(0)
        status_msg = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_msg.write(f"Processing **{uploaded_file.name}**…")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                if use_llamaparse and llama_key:
                    docs = parse_with_llamaparse(tmp_path, llama_key)
                    parser_used = "LlamaParse"
                else:
                    if use_llamaparse and not llama_key:
                        st.warning(
                            f"No LlamaCloud key — using PyPDF for **{uploaded_file.name}**."
                        )
                    docs = parse_with_fallback(tmp_path)
                    parser_used = "PyPDF"

                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["parser"] = parser_used

                new_docs.extend(docs)

                if uploaded_file.name not in st.session_state.ingested_docs:
                    st.session_state.ingested_docs.append(uploaded_file.name)

                st.success(f"✓ {uploaded_file.name} — {len(docs)} pages via {parser_used}")

            except Exception as exc:
                st.error(f"Error processing {uploaded_file.name}: {exc}")
            finally:
                os.unlink(tmp_path)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        if new_docs:
            status_msg.write("Building vector embeddings…")
            st.session_state.all_docs.extend(new_docs)
            st.session_state.vectorstore = ingest_documents(new_docs)
            status_msg.write(
                f"✅ Vector store ready — {len(new_docs)} chunks from "
                f"{len(uploaded_files)} document(s)."
            )

            with st.expander("Preview first extracted chunk"):
                preview = new_docs[0]
                st.markdown(f"**Source:** {preview.metadata.get('source')} | "
                            f"**Parser:** {preview.metadata.get('parser')}")
                st.text(preview.page_content[:600])

# ─── Tab 2: Query ────────────────────────────────────────────────────────────
with tab_query:
    st.header("Document Q&A")
    st.write("Ask questions grounded in your ingested documents. Answers cite sources.")

    if st.session_state.vectorstore is None:
        st.info("No documents ingested yet. Use the Ingest tab first.")
    else:
        query = st.text_input(
            "Your question",
            placeholder="What are the key performance benchmarks reported in the paper?",
        )
        top_k = st.slider("Source chunks to retrieve", 2, 10, 4)

        if st.button("Ask", type="primary") and query:
            from chains import get_llm, build_qa_chain
            from guardrails import rewrite_jargon

            with st.spinner("Retrieving context…"):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": top_k}
                )
                source_docs = retriever.invoke(query)

            st.subheader("Answer")
            llm = get_llm(anthropic_key)
            chain = build_qa_chain(st.session_state.vectorstore, llm, top_k)

            answer_placeholder = st.empty()
            full_answer = ""
            for chunk in chain.stream(query):
                full_answer += chunk
                answer_placeholder.markdown(full_answer)

            cleaned, changes = rewrite_jargon(full_answer)
            if changes:
                with st.expander(
                    f"🛡️ Jargon Guard — {len(changes)} term(s) rewritten in answer"
                ):
                    st.markdown("**Normalized version:**")
                    st.info(cleaned)
                    for c in changes:
                        st.write(f"• `{c['original']}` → **{c['replacement']}**")

            with st.expander("📎 Retrieved Source Chunks"):
                for i, doc in enumerate(source_docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**[{i}] {src} — page {page}**")
                    st.text(doc.page_content[:300])
                    st.divider()

# ─── Tab 3: Executive Brief ──────────────────────────────────────────────────
with tab_brief:
    st.header("Executive Business Brief")
    st.write(
        "Generate a structured, CSO-level strategic brief including a Risk/Opportunity "
        "Matrix and quarterly Implementation Roadmap — grounded in your documents."
    )

    if st.session_state.vectorstore is None:
        st.info("No documents ingested yet. Use the Ingest tab first.")
    else:
        topic = st.text_input(
            "Strategic topic or question",
            placeholder="What is the competitive landscape for transformer-based LLMs in enterprise?",
        )

        col_aud, col_depth = st.columns(2)
        with col_aud:
            audience = st.selectbox(
                "Executive Audience",
                ["C-Suite / Board", "Chief Strategy Officer", "Chief Technology Officer", "Product Leadership"],
            )
        with col_depth:
            depth = st.selectbox(
                "Brief Depth",
                ["High-level Overview", "Detailed Analysis", "Comprehensive Strategic Plan"],
            )

        if st.button("Generate Executive Brief", type="primary") and topic:
            from chains import get_llm, build_executive_summary_chain
            from guardrails import rewrite_jargon

            with st.spinner("Retrieving strategic context…"):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 6}
                )
                context_docs = retriever.invoke(topic)
                context_text = "\n\n---\n\n".join(
                    f"[Source: {d.metadata.get('source', 'doc')}]\n{d.page_content}"
                    for d in context_docs
                )

            st.subheader("Strategic Executive Brief")
            llm = get_llm(anthropic_key)
            chain = build_executive_summary_chain(llm)

            brief_placeholder = st.empty()
            full_brief = ""
            for chunk in chain.stream(
                {"topic": topic, "context": context_text, "audience": audience, "depth": depth}
            ):
                full_brief += chunk
                brief_placeholder.markdown(full_brief)

            cleaned_brief, changes = rewrite_jargon(full_brief)
            if changes:
                st.divider()
                with st.expander(
                    f"🛡️ Jargon Guard — {len(changes)} strategic term(s) normalized"
                ):
                    for c in changes:
                        st.write(f"• `{c['original']}` → **{c['replacement']}**")
                    st.markdown("**Normalized Brief:**")
                    st.markdown(cleaned_brief)

# ─── Tab 4: Guardrails ───────────────────────────────────────────────────────
with tab_guard:
    st.header("Guardrails")
    st.write(
        "Two production guardrail systems: programmatic jargon rewriting and "
        "cross-document contradiction detection."
    )

    sub_jargon, sub_contradict = st.tabs(["✏️ Jargon Rewriter", "⚠️ Contradiction Detector"])

    with sub_jargon:
        st.subheader("Jargon Rewriting Pipeline")
        st.write(
            "Automatically converts aggressive market language into preferred "
            "strategic terminology before it reaches stakeholders."
        )

        col_map, col_demo = st.columns([1, 2])
        with col_map:
            st.caption("Replacement Map")
            jargon_preview = {
                "disrupt*": "innovate",
                "dominate*": "lead the market",
                "crush*": "outperform",
                "aggressive*": "strategic",
                "war": "competitive landscape",
                "arms race": "capability investment",
                "capture the market": "grow market share",
                "wipe out": "outcompete",
                "unleash": "deploy",
                "battle": "compete",
            }
            for k, v in jargon_preview.items():
                st.write(f"`{k}` → **{v}**")

        with col_demo:
            sample = st.text_area(
                "Input text",
                value=(
                    "Our aggressive strategy will disrupt the market and dominate competitors. "
                    "We will crush legacy players and capture the market before they can respond. "
                    "This arms race for AI capabilities will determine who dominates the next decade."
                ),
                height=130,
            )

            if st.button("Rewrite Jargon", type="primary"):
                from guardrails import rewrite_jargon

                cleaned, changes = rewrite_jargon(sample)

                if changes:
                    st.success(f"Rewrote {len(changes)} term(s)")
                    st.markdown("**Rewritten Text:**")
                    st.info(cleaned)
                    st.markdown("**Changes:**")
                    for c in changes:
                        st.write(f"• `{c['original']}` → **{c['replacement']}**")
                        st.caption(f"  Context: …{c['context']}…")
                else:
                    st.success("No aggressive jargon detected.")
                    st.write(cleaned)

    with sub_contradict:
        st.subheader("Cross-Document Contradiction Detector")
        st.write(
            "Flags conflicting numerical claims across ingested source documents "
            "by identifying chunks with high semantic keyword overlap but divergent values."
        )

        if not st.session_state.all_docs:
            st.info("Ingest at least two documents to enable contradiction detection.")
        elif len(st.session_state.ingested_docs) < 2:
            st.info(
                f"Only **{st.session_state.ingested_docs[0]}** ingested. "
                "Add a second document to compare."
            )
        else:
            doc_count = len(st.session_state.ingested_docs)
            chunk_count = len(st.session_state.all_docs)
            st.write(
                f"Analyzing **{doc_count} documents** ({chunk_count} chunks) "
                "for numerical contradictions…"
            )

            if st.button("Run Contradiction Detection", type="primary"):
                from guardrails import detect_contradictions

                with st.spinner("Scanning…"):
                    contradictions = detect_contradictions(st.session_state.all_docs)

                if contradictions:
                    st.error(f"⚠️ Found {len(contradictions)} potential contradiction(s)")

                    for i, c in enumerate(contradictions, 1):
                        icon = "🔴" if c["severity"] == "high" else "🟡"
                        header = (
                            f"{icon} #{i} — **{c['source_a']}** vs **{c['source_b']}** | "
                            f"Values: {c['value_a']} vs {c['value_b']} "
                            f"({c['difference_pct']}% apart)"
                        )
                        with st.expander(header):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**{c['source_a']}**")
                                st.text(c["text_a"])
                                st.metric("Reported value", c["value_a"])
                            with col_b:
                                st.markdown(f"**{c['source_b']}**")
                                st.text(c["text_b"])
                                st.metric("Reported value", c["value_b"])
                            st.caption(
                                f"Keyword overlap: {c['keyword_overlap']:.0%} | "
                                f"Divergence: {c['difference_pct']}% | "
                                f"Severity: {c['severity']}"
                            )
                else:
                    st.success("✅ No significant contradictions detected across documents.")
