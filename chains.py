from typing import List

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

OLLAMA_MODEL = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest"


def get_llm(api_key: str = "") -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_predict=2048,
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


def build_executive_summary_chain(llm: ChatOllama):
    """LCEL chain for structured executive-level strategic briefs."""
    prompt = ChatPromptTemplate.from_template(
        """You are a Chief Strategy Officer producing a structured executive business brief.
Audience: {audience}
Depth: {depth}

Using the research context below, produce a comprehensive strategic brief on the given topic.
Be decisive, data-grounded, and boardroom-ready.

TOPIC: {topic}

RESEARCH CONTEXT:
{context}

---

Generate the brief using this exact structure:

## Executive Summary
[2-3 sentence strategic summary for leadership]

## Market & Competitive Landscape
[Key findings and dynamics from the research]

## Strategic Opportunity Analysis
[3-4 clearly defined opportunities with supporting evidence]

## Risk / Opportunity Matrix

| Factor | Type | Probability | Impact | Strategic Response |
|--------|------|-------------|--------|--------------------|
[5 rows - mix of risks and opportunities]

## Recommended Strategic Initiatives
1. [Initiative Name] - [Description and rationale]
2. [Initiative Name] - [Description and rationale]
3. [Initiative Name] - [Description and rationale]

## Key Performance Indicators
- [KPI 1]: [Definition and target]
- [KPI 2]: [Definition and target]
- [KPI 3]: [Definition and target]

## Implementation Roadmap
| Quarter | Focus | Key Deliverables |
|---------|-------|-----------------|
| Q1 | | |
| Q2 | | |
| Q3 | | |
| Q4 | | |

## Strategic Recommendation
[1-2 sentence decisive recommendation for leadership action]
"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain
