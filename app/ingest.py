"""
ingest.py — Ingestion pipeline: load PDFs -> chunk -> embed + upsert to Qdrant.

Run as a module from the project root:
    python -m app.ingest --pdf-dir data/pdfs
    python -m app.ingest --pdf-dir data/pdfs --collection rag_docs --recreate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .embeddings import load_dense_model, load_sparse_model
from .vector_store import ensure_collection, get_client, reset_embedded_store, upsert_chunks


# --------------------------------------------------------------------------- #
# Load + chunk
# --------------------------------------------------------------------------- #

def load_pdf_paths(pdf_paths: list[Path]) -> list:
    """Load specific PDF files into per-page LangChain Documents.

    PyPDFLoader attaches ``source`` (path) and ``page`` (0-indexed) metadata,
    which we carry through to Qdrant for citations.
    """
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    for path in pdf_paths:
        try:
            pages = PyPDFLoader(str(path)).load()
        except Exception as exc:  # noqa: BLE001 — skip unreadable PDFs, keep going
            print(f"  ! skipping {path.name}: {exc}")
            continue
        for page in pages:  # normalise source to just the filename
            page.metadata["source"] = path.name
        docs.extend(pages)
        print(f"  + loaded {path.name} ({len(pages)} pages)")

    print(f"Loaded {len(pdf_paths)} PDF(s) -> {len(docs)} pages")
    return docs


def load_pdfs(pdf_dir: Path) -> list:
    """Load every PDF in ``pdf_dir``."""
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir.resolve()}")
    return load_pdf_paths(pdf_paths)


def split_documents(docs: list) -> list:
    """Split page Documents into overlapping chunks and assign a chunk_id."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    # Stable, human-readable chunk id: "<source>::p<page>::c<n-on-page>".
    per_page_counter: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = int(chunk.metadata.get("page", -1))
        key = (source, page)
        n = per_page_counter.get(key, 0)
        per_page_counter[key] = n + 1
        chunk.metadata["chunk_id"] = f"{source}::p{page}::c{n}"

    print(f"Split into {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into a Qdrant hybrid index.")
    parser.add_argument("--pdf-dir", default=str(config.DATA_DIR),
                        help=f"Directory of PDFs (default: {config.DATA_DIR})")
    parser.add_argument("--collection", default=config.DEFAULT_COLLECTION,
                        help=f"Qdrant collection name (default: {config.DEFAULT_COLLECTION})")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the collection before ingesting")
    args = parser.parse_args()

    docs = load_pdfs(Path(args.pdf_dir))
    chunks = split_documents(docs)

    dense_model = load_dense_model()
    sparse_model = load_sparse_model()

    # For embedded mode, wipe the folder before opening the client so --recreate
    # is a real clean slate (delete_collection alone doesn't purge local storage).
    if args.recreate:
        reset_embedded_store()
    client = get_client()

    ensure_collection(client, args.collection, recreate=args.recreate)
    n_indexed = upsert_chunks(client, args.collection, chunks, dense_model, sparse_model)

    count = client.count(args.collection, exact=True).count
    print("-" * 48)
    print(f"{n_indexed} chunks indexed. "
          f"(collection '{args.collection}' now holds {count} points)")


if __name__ == "__main__":
    main()
