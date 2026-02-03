"""Document ingestion example for the Advanced RAG Pipeline.

This example demonstrates various ways to ingest documents into the pipeline.
"""

import asyncio
from pathlib import Path

from rag_interface import AdvancedRAGPipeline


async def ingest_single_document():
    """Ingest a single document."""
    pipeline = AdvancedRAGPipeline()

    result = await pipeline.ingest(
        content="Python is a high-level programming language known for its simplicity.",
        source="python_intro.txt",
        collection_name="programming",
        chunk_size=256,
        chunk_overlap=25,
        metadata={"language": "python", "level": "beginner"},
    )

    print(f"Single document ingested:")
    print(f"  - Document ID: {result.document_id}")
    print(f"  - Chunks: {result.chunk_count}")
    print(f"  - Characters: {result.total_characters}")
    print(f"  - Processing time: {result.processing_time_ms:.2f}ms")


async def ingest_multiple_documents():
    """Ingest multiple documents in batch."""
    pipeline = AdvancedRAGPipeline()

    documents = [
        (
            "JavaScript is a versatile programming language primarily used for web development.",
            "js_intro.txt",
            {"language": "javascript", "level": "beginner"},
        ),
        (
            "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.",
            "ts_intro.txt",
            {"language": "typescript", "level": "intermediate"},
        ),
        (
            "Rust is a systems programming language focused on safety and performance.",
            "rust_intro.txt",
            {"language": "rust", "level": "advanced"},
        ),
    ]

    results = await pipeline.ingest_batch(
        documents=documents,
        collection_name="programming",
    )

    print(f"\nBatch ingestion results ({len(results)} documents):")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.document_id}: {result.chunk_count} chunks")


async def ingest_from_file():
    """Ingest a document from a file (example)."""
    pipeline = AdvancedRAGPipeline()

    # Example: Read from a file
    sample_content = """
    This is an example of ingesting content from a file.
    In a real application, you would read the file content like this:

    with open('document.txt', 'r') as f:
        content = f.read()

    Then ingest it into the pipeline.
    """

    result = await pipeline.ingest(
        content=sample_content,
        source="example_file.txt",
        metadata={"source_type": "file"},
    )

    print(f"\nFile ingestion result:")
    print(f"  - Document ID: {result.document_id}")
    print(f"  - Collection: {result.collection_name}")


async def main():
    print("=== Document Ingestion Examples ===\n")

    await ingest_single_document()
    await ingest_multiple_documents()
    await ingest_from_file()


if __name__ == "__main__":
    asyncio.run(main())
