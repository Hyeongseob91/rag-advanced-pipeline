"""Custom LLM configuration example for the Advanced RAG Pipeline.

This example demonstrates how to use custom LLM adapters with the pipeline.
"""

import asyncio

from rag_interface import AdvancedRAGPipeline
from rag_infra.llm import OpenAIAdapter, VLLMAdapter, OllamaAdapter


async def use_openai():
    """Use OpenAI as the LLM provider."""
    print("=== Using OpenAI ===")

    # Create custom OpenAI adapter
    llm = OpenAIAdapter(
        api_key="your-api-key",  # Replace with actual key
        model="gpt-4o",
    )

    pipeline = AdvancedRAGPipeline(llm_adapter=llm)

    # Now queries will use OpenAI
    # result = await pipeline.query("What is AI?")
    print("OpenAI adapter configured successfully")


async def use_vllm():
    """Use vLLM (GPU server) as the LLM provider."""
    print("\n=== Using vLLM (GPU) ===")

    # Create vLLM adapter pointing to your GPU server
    llm = VLLMAdapter(
        base_url="http://localhost:8000/v1",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        # api_key="optional-key",
    )

    pipeline = AdvancedRAGPipeline(llm_adapter=llm)

    print("vLLM adapter configured for GPU inference")
    print(f"Model: {llm.model_name}")


async def use_ollama():
    """Use Ollama (local) as the LLM provider."""
    print("\n=== Using Ollama (Local) ===")

    # Create Ollama adapter for local inference
    llm = OllamaAdapter(
        base_url="http://localhost:11434",
        model="llama3.1:8b",
    )

    pipeline = AdvancedRAGPipeline(llm_adapter=llm)

    print("Ollama adapter configured for local inference")
    print(f"Model: {llm.model_name}")


async def combine_custom_adapters():
    """Combine multiple custom adapters."""
    print("\n=== Combining Custom Adapters ===")

    from rag_infra.embedding import OpenAIEmbeddingAdapter
    from rag_infra.vectordb import WeaviateAdapter
    from rag_infra.chunking import SemanticChunker

    # Create custom adapters for each component
    llm = VLLMAdapter(
        base_url="http://gpu-server:8000/v1",
        model="llama-3-70b",
    )

    embedding = OpenAIEmbeddingAdapter(
        api_key="your-api-key",
        model="text-embedding-3-small",
    )

    vectordb = WeaviateAdapter(
        host="weaviate-server",
        port=8080,
    )

    chunker = SemanticChunker(
        embedding_port=embedding,
        breakpoint_threshold=0.5,
    )

    # Create pipeline with all custom adapters
    pipeline = AdvancedRAGPipeline(
        llm_adapter=llm,
        embedding_adapter=embedding,
        vectordb_adapter=vectordb,
        chunker_adapter=chunker,
    )

    print("Pipeline configured with all custom adapters:")
    print(f"  - LLM: vLLM (GPU)")
    print(f"  - Embedding: OpenAI")
    print(f"  - VectorDB: Weaviate")
    print(f"  - Chunker: Semantic")


async def main():
    print("=== Custom LLM Configuration Examples ===\n")

    await use_openai()
    await use_vllm()
    await use_ollama()
    await combine_custom_adapters()


if __name__ == "__main__":
    asyncio.run(main())
