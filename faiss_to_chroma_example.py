"""Examples for indexing transcript data with metadata (including `np`) in FAISS and Chroma.

This script shows how to:
1) Preserve row metadata (`np`, `dummy2`, etc.) in LangChain Documents.
2) Query FAISS and retrieve `np` from matched documents.
3) Screen/filter FAISS results by `np` values (post-filtering pattern).
4) Query Chroma with metadata filters directly.

Usage:
    export OPENAI_API_KEY=...
    python faiss_to_chroma_example.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS


CSV_PATH = Path("df_combined3.csv")
CHROMA_DIR = Path("chroma3_index")
FAISS_DIR = Path("faiss3_index")
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def _normalize_metadata_value(value: Any) -> Any:
    """Keep metadata JSON-friendly while preserving numeric fields for filtering."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, float, bool, str)):
        return value

    try:
        as_float = float(value)
        if as_float.is_integer():
            return int(as_float)
        return as_float
    except (TypeError, ValueError):
        return str(value)


def _sanitize_text_for_embedding(text: Any) -> str:
    """Normalize text to UTF-8-safe string to avoid malformed JSON payloads."""
    safe_text = str(text).replace("\x00", " ")
    # Drop invalid unicode surrogates that can break JSON serialization downstream.
    safe_text = safe_text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return safe_text.strip()


def build_documents_with_metadata(df: pd.DataFrame) -> list[Document]:
    """Create Document objects with all non-transcript columns, including `np`, as metadata."""
    documents: list[Document] = []

    for row_id, row in df.iterrows():
        transcript = row.get("transcript")
        if pd.isna(transcript):
            continue

        metadata: dict[str, Any] = {"row_id": int(row_id)}

        for col in df.columns:
            if col == "transcript":
                continue
            metadata[col] = _normalize_metadata_value(row.get(col))

        clean_transcript = _sanitize_text_for_embedding(transcript)
        if not clean_transcript:
            continue

        documents.append(Document(page_content=clean_transcript, metadata=metadata))

    return documents


def add_chunk_metadata(docs: list[Document]) -> list[Document]:
    """Split documents and preserve metadata per chunk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunked_docs = splitter.split_documents(docs)

    for chunk_id, doc in enumerate(chunked_docs):
        doc.metadata["chunk_id"] = chunk_id

    return chunked_docs


def build_faiss_index(docs: list[Document], embeddings_model: OpenAIEmbeddings) -> FAISS:
    """Build and persist FAISS index with metadata-bearing Documents."""
    if not docs:
        raise ValueError("No documents to index after filtering and sanitization.")

    # Embed in batches to reduce payload size and avoid API request failures.
    vectordb = FAISS.from_documents(documents=docs[:BATCH_SIZE], embedding=embeddings_model)
    for i in range(BATCH_SIZE, len(docs), BATCH_SIZE):
        vectordb.add_documents(docs[i : i + BATCH_SIZE])

    vectordb.save_local(str(FAISS_DIR))
    return vectordb


def load_faiss_index(embeddings_model: OpenAIEmbeddings) -> FAISS:
    """Load persisted FAISS index."""
    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings_model,
        allow_dangerous_deserialization=True,
    )


def search_faiss_with_np_filter(
    query_text: str,
    db_index: FAISS,
    k: int = 2,
    np_equals: Any | None = None,
    np_min: float | None = None,
    np_max: float | None = None,
    fetch_k: int = 40,
) -> list[Document]:
    """Search FAISS and optionally filter by metadata `np` (post-filtering)."""
    # FAISS in LangChain retrieves nearest neighbors first, then we can filter on metadata.
    # Fetch more candidates to make filtering effective.
    candidates = db_index.similarity_search_with_score(query_text, k=fetch_k)

    filtered: list[Document] = []
    for doc, _score in candidates:
        np_value = doc.metadata.get("np")

        if np_equals is not None and np_value != np_equals:
            continue

        if np_min is not None:
            try:
                if float(np_value) < np_min:
                    continue
            except (TypeError, ValueError):
                continue

        if np_max is not None:
            try:
                if float(np_value) > np_max:
                    continue
            except (TypeError, ValueError):
                continue

        filtered.append(doc)
        if len(filtered) >= k:
            break

    return filtered


def build_chroma_index(docs: list[Document], embeddings_model: OpenAIEmbeddings) -> Chroma:
    """Build and persist Chroma index with metadata-bearing Documents."""
    if not docs:
        raise ValueError("No documents to index after filtering and sanitization.")

    vectordb = Chroma(
        embedding_function=embeddings_model,
        persist_directory=str(CHROMA_DIR),
        collection_name="df_combined3_transcripts",
    )
    for i in range(0, len(docs), BATCH_SIZE):
        vectordb.add_documents(docs[i : i + BATCH_SIZE])

    vectordb.persist()
    return vectordb


def search_chroma_with_np_filter(query_text: str, db_index: Chroma, k: int = 2, np_equals: Any | None = None) -> list[Document]:
    """Search Chroma and filter by `np` using native metadata filter."""
    where = {"np": np_equals} if np_equals is not None else None
    return db_index.similarity_search(query_text, k=k, filter=where)


def print_results(results: list[Document], title: str) -> None:
    print(f"\n{title}\n")
    if not results:
        print("No results found for current filter.\n")
        return

    for i, doc in enumerate(results, 1):
        np_value = doc.metadata.get("np")
        print(f"Result {i} | np={np_value} | metadata={doc.metadata}")
        print(f"{doc.page_content}\n")


def main() -> None:
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    df_combined3 = pd.read_csv(CSV_PATH)

    # Keep your original filtering.
    filtered_df = df_combined3[df_combined3["dummy2"] == 1]

    documents = build_documents_with_metadata(filtered_df)
    chunked_docs = add_chunk_metadata(documents)

    faiss_db = build_faiss_index(chunked_docs, embeddings_model)
    _ = build_chroma_index(chunked_docs, embeddings_model)

    print(f"FAISS vectors: {faiss_db.index.ntotal}")

    query = "문재인 대통령에게 가장 비판적인 기사는? 기사가 나온 신문은?"

    # Reload FAISS and demonstrate retrieving np.
    loaded_faiss = load_faiss_index(embeddings_model)
    faiss_results = search_faiss_with_np_filter(query, loaded_faiss, k=2)
    print_results(faiss_results, "FAISS results (retrieve np metadata)")

    # Example: screen out by np value (exact match).
    faiss_np_filtered = search_faiss_with_np_filter(query, loaded_faiss, k=2, np_equals=1)
    print_results(faiss_np_filtered, "FAISS results filtered by np == 1")


if __name__ == "__main__":
    main()
