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
    st.session_state.ingested_docs = []
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []
# Per-level: {response, judge (parsed dict), improved}
for _lvl in range(1, 5):
    if f"level_{_lvl}" not in st.session_state:
        st.session_state[f"level_{_lvl}"] = {}

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
st.caption("LlamaParse · ChromaDB · LangChain · Llama 3.2 via Ollama · Judge AI")

tab_ingest, tab_query, tab_learn, tab_guard = st.tabs(
    ["📥 Ingest Documents", "🔍 Query", "📚 Learning Levels", "🛡️ Guardrails"]
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

# ─── Tab 3: Learning Levels ──────────────────────────────────────────────────
with tab_learn:
    from chains import LEVEL_CONFIGS

    st.header("Progressive Learning Levels")
    st.write(
        "Get 4 progressively deeper explanations of your document — each one building on "
        "the last. A judge LLM automatically evaluates every response and triggers an "
        "improvement pass when quality falls below 7/10."
    )

    # Level overview cards
    cols = st.columns(4)
    for i, col in enumerate(cols, start=1):
        cfg = LEVEL_CONFIGS[i]
        col.metric(
            label=f"{cfg['icon']} Level {i}",
            value=cfg["name"],
            delta=cfg["description"],
            delta_color="off",
        )

    st.divider()

    if st.session_state.vectorstore is None:
        st.info("No documents ingested yet. Use the Ingest tab first.")
    else:
        topic_focus = st.text_input(
            "Focus topic (optional)",
            placeholder="e.g. 'attention mechanism' — leave blank to cover the full document",
            key="level_topic",
        )

        lev_tabs = st.tabs(
            [f"{LEVEL_CONFIGS[i]['icon']} Level {i} — {LEVEL_CONFIGS[i]['name']}" for i in range(1, 5)]
        )

        for level, lev_tab in enumerate(lev_tabs, start=1):
            with lev_tab:
                cfg = LEVEL_CONFIGS[level]
                st.subheader(f"Level {level}: {cfg['name']}")
                st.caption(cfg["description"])

                state_key = f"level_{level}"
                data = st.session_state[state_key]

                # ── Display stored results (from previous generation) ──────────
                if data.get("response"):
                    display_text = data.get("improved") or data["response"]
                    st.markdown(display_text)

                    if data.get("judge"):
                        judge = data["judge"]
                        score = judge.get("score", 0)
                        verdict = judge.get("verdict", "UNKNOWN")
                        score_icon = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"

                        with st.expander(
                            f"🧑‍⚖️ Judge Evaluation — {score_icon} {score}/10 ({verdict})"
                        ):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Score", f"{score}/10")
                                st.write(f"**Accuracy:** {judge.get('accuracy', '—')}")
                                st.write(f"**Depth Fit:** {judge.get('depth_fit', '—')}")
                            with col_b:
                                st.write(f"**Clarity:** {judge.get('clarity', '—')}")
                                st.write(f"**Missing:** {judge.get('missing', '—')}")
                            st.info(f"**Feedback:** {judge.get('feedback', '—')}")

                    if data.get("improved"):
                        st.success("✨ Response was automatically improved by judge feedback.")

                    st.divider()

                # ── Generate / Regenerate button ─────────────────────────────
                btn_label = (
                    f"Regenerate Level {level}"
                    if data.get("response")
                    else f"Generate Level {level}"
                )
                if st.button(btn_label, key=f"btn_level_{level}", type="primary"):
                    from chains import (
                        get_llm, get_judge_llm,
                        build_level_chain, build_judge_chain, parse_judge_output,
                    )

                    st.session_state[state_key] = {}  # clear old results

                    # Retrieve document context
                    with st.spinner("Retrieving document context…"):
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 6}
                        )
                        query_text = topic_focus if topic_focus else "main topic and key concepts"
                        context_docs = retriever.invoke(query_text)
                        context_text = "\n\n---\n\n".join(
                            f"[Source: {d.metadata.get('source', 'doc')}]\n{d.page_content}"
                            for d in context_docs
                        )

                    llm = get_llm()
                    judge_llm = get_judge_llm()

                    # ── Stream the level response ─────────────────────────────
                    st.subheader(f"Level {level} Response")
                    resp_placeholder = st.empty()
                    full_response = ""
                    level_chain = build_level_chain(level, llm)
                    for chunk in level_chain.stream({
                        "context": context_text,
                        "topic": topic_focus or "the main topic of this document",
                    }):
                        full_response += chunk
                        resp_placeholder.markdown(full_response)

                    st.session_state[state_key]["response"] = full_response

                    # ── Judge evaluation ──────────────────────────────────────
                    with st.spinner("🧑‍⚖️ Judge evaluating response quality…"):
                        judge_chain = build_judge_chain(judge_llm)
                        judge_raw = judge_chain.invoke({
                            "level": level,
                            "level_name": cfg["name"],
                            "level_description": cfg["description"],
                            "criteria": cfg["judge_criteria"],
                            "response": full_response,
                            "context": context_text,
                        })
                        parsed_judge = parse_judge_output(judge_raw)

                    st.session_state[state_key]["judge"] = parsed_judge

                    score = parsed_judge.get("score", 0)
                    verdict = parsed_judge.get("verdict", "FAIL")
                    score_icon = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"

                    with st.expander(
                        f"🧑‍⚖️ Judge Evaluation — {score_icon} {score}/10 ({verdict})"
                    ):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Score", f"{score}/10")
                            st.write(f"**Accuracy:** {parsed_judge.get('accuracy', '—')}")
                            st.write(f"**Depth Fit:** {parsed_judge.get('depth_fit', '—')}")
                        with col_b:
                            st.write(f"**Clarity:** {parsed_judge.get('clarity', '—')}")
                            st.write(f"**Missing:** {parsed_judge.get('missing', '—')}")
                        st.info(f"**Feedback:** {parsed_judge.get('feedback', '—')}")

                    # ── Auto-improve if score < 7 ─────────────────────────────
                    if verdict == "FAIL":
                        st.warning(
                            f"Score {score}/10 is below threshold — generating improved response…"
                        )
                        improved_chain = build_level_chain(
                            level, llm,
                            feedback=parsed_judge.get("feedback", ""),
                        )
                        st.subheader("✨ Improved Response")
                        imp_placeholder = st.empty()
                        improved_response = ""
                        for chunk in improved_chain.stream({
                            "context": context_text,
                            "topic": topic_focus or "the main topic of this document",
                        }):
                            improved_response += chunk
                            imp_placeholder.markdown(improved_response)

                        st.session_state[state_key]["improved"] = improved_response
                    else:
                        st.success(f"✅ Score {score}/10 — Response passed quality check.")

                    st.rerun()

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
