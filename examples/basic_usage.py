"""Basic usage example for the Advanced RAG Pipeline.

This example demonstrates the simplest way to use the pipeline
with default configuration from environment variables.
"""

import asyncio

from rag_interface import AdvancedRAGPipeline


async def main():
    # Create pipeline with default settings (from environment)
    pipeline = AdvancedRAGPipeline()

    # Ingest a sample document
    print("Ingesting document...")
    ingest_result = await pipeline.ingest(
        content="""
        Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly
        programmed. It focuses on developing computer programs that can access
        data and use it to learn for themselves.

        The process begins with observations or data, such as examples, direct
        experience, or instruction, to look for patterns in data and make better
        decisions in the future. The primary aim is to allow computers to learn
        automatically without human intervention.

        There are several types of machine learning:
        1. Supervised Learning: The algorithm learns from labeled training data
        2. Unsupervised Learning: The algorithm finds patterns in unlabeled data
        3. Reinforcement Learning: The algorithm learns through trial and error
        """,
        source="ml_intro.txt",
        metadata={"topic": "machine_learning", "type": "introduction"},
    )
    print(f"Ingested document with {ingest_result.chunk_count} chunks")

    # Query the pipeline
    print("\nQuerying pipeline...")
    result = await pipeline.query(
        question="What is machine learning and what are its types?",
        top_k=3,
    )

    print(f"\nAnswer: {result.answer}")
    print(f"\nSources ({len(result.sources)}):")
    for source in result.sources:
        print(f"  - Score: {source.score:.3f} | {source.content[:100]}...")

    print(f"\nMetrics:")
    print(f"  - Model: {result.model}")
    print(f"  - Tokens: {result.total_tokens}")
    print(f"  - Retrieval time: {result.retrieval_time_ms:.2f}ms")
    print(f"  - Generation time: {result.generation_time_ms:.2f}ms")
    print(f"  - Total time: {result.total_time_ms:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
