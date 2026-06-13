import os
from typing import List

from langchain.schema import Document


def parse_with_llamaparse(file_path: str, api_key: str) -> List[Document]:
    """Parse PDF with LlamaParse, preserving tables and mathematical equations."""
    try:
        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            parsing_instruction=(
                "Preserve all tables in markdown format. "
                "Render mathematical equations in LaTeX notation using $...$ for inline "
                "and $$...$$ for block equations. "
                "Maintain document structure including headers and section hierarchy. "
                "Keep figure captions, footnotes, and citation references intact."
            ),
            verbose=False,
        )

        llama_docs = parser.load_data(file_path)

        if not llama_docs:
            return parse_with_fallback(file_path)

        langchain_docs = []
        for i, doc in enumerate(llama_docs):
            langchain_docs.append(
                Document(
                    page_content=doc.text,
                    metadata={
                        "page": i + 1,
                        "parser": "llamaparse",
                        "file_path": file_path,
                    },
                )
            )

        return langchain_docs

    except Exception as e:
        print(f"LlamaParse failed ({e}). Falling back to PyPDF.")
        return parse_with_fallback(file_path)


def parse_with_fallback(file_path: str) -> List[Document]:
    """Parse PDF with PyPDF (free fallback, no table/equation preservation)."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["parser"] = "pypdf"

    return docs
