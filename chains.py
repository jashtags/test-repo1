import re
from typing import Dict, List

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

OLLAMA_MODEL = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest"

# Four progressive learning levels — each builds on the previous one.
LEVEL_CONFIGS = {
    1: {
        "name": "Foundations",
        "icon": "🌱",
        "description": "Complete beginner — no prior knowledge assumed",
        "system": (
            "You are a patient, enthusiastic educator explaining a topic to someone who knows NOTHING about it. "
            "Assume zero technical background. Define every term before using it. "
            "Use everyday analogies and real-world comparisons to make abstract ideas concrete. "
            "Explain WHY this topic matters in practical, everyday terms. "
            "Cover: what this is, why it was created, the main types or variants, and real-world use cases. "
            "Keep sentences short and use bullet points and simple headers. "
            "Never use unexplained jargon — replace every technical term with a plain-English explanation first."
        ),
        "task": (
            "Provide a complete beginner-friendly background on the topic covered in these documents. "
            "Explain everything from scratch assuming the reader has never encountered any of these concepts before."
        ),
        "judge_criteria": (
            "1. No unexplained jargon — every technical term is defined in plain English\n"
            "2. Uses at least 2 real-world analogies or everyday comparisons\n"
            "3. Covers: what it is, why it exists, main types/variants, and practical use cases\n"
            "4. Accessible to a curious person with zero technical background"
        ),
    },
    2: {
        "name": "Concepts & Mechanisms",
        "icon": "⚙️",
        "description": "Builds on Level 1 — explains HOW things work",
        "system": (
            "You are an educator teaching someone who has just finished a beginner introduction. "
            "They already know: basic definitions, what the technology is, why it exists, and general use cases. "
            "Build on that knowledge — do NOT re-explain basics they already learned. "
            "Explain HOW things work internally: mechanisms, components, data flow, and interactions. "
            "Introduce technical terms but explain each one clearly in context. "
            "Explain key design choices: why was it built this way? What are the main trade-offs? "
            "Cover: internal architecture, key algorithms at a conceptual level, how components interact, and design rationale."
        ),
        "task": (
            "Explain HOW the concepts in these documents work — mechanisms, internal architecture, "
            "and design decisions. Assume the reader knows the basics but is ready for the technical 'how'."
        ),
        "judge_criteria": (
            "1. Does NOT re-explain basic definitions (assumes Level 1 knowledge)\n"
            "2. Explains internal mechanisms and how components interact\n"
            "3. Covers design choices and their rationale\n"
            "4. Appropriate for someone with basic familiarity wanting to understand the 'how'"
        ),
    },
    3: {
        "name": "Technical Deep Dive",
        "icon": "🔬",
        "description": "Builds on Levels 1 & 2 — detailed technical analysis",
        "system": (
            "You are a senior engineer teaching someone who understands both the concepts and the mechanisms. "
            "They know: what it is, why it exists, how components connect, and key design choices. "
            "Go deep into technical specifics: mathematical intuition, specific algorithms, implementation details. "
            "Compare with alternative approaches — what specific trade-offs were made and why? "
            "Discuss limitations, edge cases, and known failure modes with concrete specifics. "
            "Reference concrete numbers, benchmarks, or metrics from the document wherever possible. "
            "Cover: algorithmic details, mathematical intuition, performance trade-offs, limitations, and open problems."
        ),
        "task": (
            "Provide a detailed technical analysis — algorithms, implementation specifics, mathematical intuition, "
            "trade-offs, and limitations. Assume solid understanding of mechanisms from Levels 1 and 2."
        ),
        "judge_criteria": (
            "1. Covers specific algorithmic or mathematical details with precision\n"
            "2. References concrete numbers or benchmarks from the source documents\n"
            "3. Discusses trade-offs, limitations, and alternative approaches with specifics\n"
            "4. Appropriate depth for a practitioner or advanced student"
        ),
    },
    4: {
        "name": "Expert Paper Summary",
        "icon": "📄",
        "description": "Exact technical summary for domain experts — paper as written",
        "system": (
            "You are writing a precise expert summary of a technical paper for a researcher deciding whether to read it. "
            "Be dense and precise — no hand-holding, no analogies, no introductory scaffolding. "
            "Cover: problem statement and motivation, key contributions, methodology, experimental setup, "
            "results with exact numbers, ablation studies, limitations, and directions for future work. "
            "Explicitly note what is novel versus what builds on prior work, citing specific prior methods by name. "
            "Be critical: identify weaknesses, unstated assumptions, and open questions the paper leaves unresolved. "
            "Use exact figures (accuracy percentages, speedup factors, dataset sizes, benchmark names) from the document."
        ),
        "task": (
            "Write a complete expert-level summary — problem, contributions, methodology, results with metrics, "
            "limitations, and significance. Be precise, dense, and technical. No simplification."
        ),
        "judge_criteria": (
            "1. Covers problem statement, contributions, methodology, results, and limitations\n"
            "2. Cites specific numbers and benchmark results from the document\n"
            "3. Notes what is novel vs. what builds on prior work\n"
            "4. Dense and precise — suitable for an expert researcher or engineer"
        ),
    },
}


def get_llm(api_key: str = "") -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_predict=2048,
    )


def get_judge_llm() -> ChatOllama:
    """Judge uses temperature=0 for deterministic, consistent evaluations."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        num_predict=512,
    )


def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')} | Page: {d.metadata.get('page', '?')}]\n"
        f"{d.page_content}"
        for d in docs
    )


def build_qa_chain(vectorstore: Chroma, llm: ChatOllama, top_k: int = 4):
    """LCEL RAG chain: retrieve → format → prompt → LLM → string."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    prompt = ChatPromptTemplate.from_template(
        """You are a precise document analyst. Answer the question using only the provided context.
If the answer is not in the context, say exactly: "The documents don't contain information about this topic."
Cite the source document and page when referencing specific facts.

Context:
{context}

Question: {question}

Answer:"""
    )
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def build_level_chain(level: int, llm: ChatOllama, feedback: str = ""):
    """Build a progressive learning chain for a given level (1–4).

    If feedback is provided, the judge's improvement notes are appended to the
    system prompt so the second-pass response addresses identified gaps.
    """
    cfg = LEVEL_CONFIGS[level]
    system = cfg["system"]
    if feedback:
        system += (
            "\n\nIMPORTANT — A quality review identified these issues to fix:\n"
            + feedback
            + "\nAddress every point above in your improved response."
        )

    # Bake system text into the template so only {context} and {topic} are variables.
    prompt_text = (
        system
        + "\n\nDOCUMENT CONTEXT:\n{context}\n\n"
        + "FOCUS: {topic}\n\n"
        + f"TASK: {cfg['task']}\n\n"
        + "Response:"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return prompt | llm | StrOutputParser()


def build_judge_chain(judge_llm: ChatOllama):
    """Judge LLM that evaluates a level response and returns structured feedback."""
    prompt = ChatPromptTemplate.from_template(
        """You are an expert educational content evaluator and technical reviewer.

EVALUATING: Level {level} — {level_name} ({level_description})

QUALITY CRITERIA FOR LEVEL {level}:
{criteria}

━━━ RESPONSE TO EVALUATE ━━━
{response}
━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ DOCUMENT CONTEXT (ground truth) ━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluate strictly against the criteria. Output in EXACTLY this format (keep the labels):
SCORE: [integer 1-10]
ACCURACY: [one sentence on factual accuracy vs. source documents]
DEPTH_FIT: [one sentence on whether depth matches Level {level} requirements]
CLARITY: [one sentence on clarity and accessibility]
MISSING: [bullet list of important content that is absent]
FEEDBACK: [2-3 actionable sentences on what to fix or add]
VERDICT: [PASS if score >= 7, else FAIL]
"""
    )
    return prompt | judge_llm | StrOutputParser()


def parse_judge_output(judge_text: str) -> Dict:
    """Extract structured fields from judge LLM output."""
    result: Dict = {
        "score": 0,
        "accuracy": "",
        "depth_fit": "",
        "clarity": "",
        "missing": "",
        "feedback": "",
        "verdict": "FAIL",
        "raw": judge_text,
    }
    patterns = {
        "score": r"SCORE:\s*(\d+)",
        "accuracy": r"ACCURACY:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "depth_fit": r"DEPTH_FIT:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "clarity": r"CLARITY:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "missing": r"MISSING:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "feedback": r"FEEDBACK:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "verdict": r"VERDICT:\s*(PASS|FAIL)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, judge_text, re.DOTALL | re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            if key == "score":
                try:
                    result[key] = int(value)
                except ValueError:
                    pass
            else:
                result[key] = value
    return result
