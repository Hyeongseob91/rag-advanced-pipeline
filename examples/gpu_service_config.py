"""GPU Service configuration example for the Advanced RAG Pipeline.

This example demonstrates how to configure the pipeline to use GPU-accelerated
services like vLLM for LLM inference and Infinity for embeddings.
"""

import asyncio
import os

from rag_interface import AdvancedRAGPipeline
from rag_infra.llm import VLLMAdapter
from rag_infra.embedding import InfinityAdapter
from rag_infra.vectordb import WeaviateAdapter


def setup_environment_for_gpu():
    """Set environment variables for GPU services.

    In production, these would be set in your .env file or deployment config.
    """
    os.environ.update({
        # LLM - vLLM on GPU
        "LLM_PROVIDER": "vllm",
        "LLM_BASE_URL": "http://gpu-server:8000/v1",
        "LLM_MODEL": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",

        # Embedding - Infinity on GPU
        "EMBEDDING_PROVIDER": "infinity",
        "EMBEDDING_BASE_URL": "http://gpu-server:7997",
        "EMBEDDING_MODEL": "BAAI/bge-m3",
        "EMBEDDING_DIMENSION": "1024",

        # VectorDB
        "VECTORDB_PROVIDER": "weaviate",
        "VECTORDB_HOST": "weaviate",
        "VECTORDB_PORT": "8080",
    })


async def create_gpu_pipeline_from_env():
    """Create a GPU-accelerated pipeline using environment configuration."""
    print("=== GPU Pipeline from Environment ===")

    # Set up environment (in production, this comes from .env)
    setup_environment_for_gpu()

    # Pipeline will use GPU services based on environment
    pipeline = AdvancedRAGPipeline()

    print("Pipeline configured from environment:")
    print(f"  - LLM Provider: {pipeline.settings.llm.provider.value}")
    print(f"  - LLM Model: {pipeline.settings.llm.model}")
    print(f"  - Embedding Provider: {pipeline.settings.embedding.provider.value}")
    print(f"  - Embedding Model: {pipeline.settings.embedding.model}")


async def create_gpu_pipeline_programmatic():
    """Create a GPU-accelerated pipeline programmatically."""
    print("\n=== GPU Pipeline Programmatic ===")

    # Create GPU-specific adapters
    llm = VLLMAdapter(
        base_url="http://gpu-server:8000/v1",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        timeout=120.0,  # Longer timeout for large models
    )

    embedding = InfinityAdapter(
        base_url="http://gpu-server:7997",
        model="BAAI/bge-m3",
        dimension=1024,
    )

    vectordb = WeaviateAdapter(
        host="weaviate",
        port=8080,
        default_collection="gpu_documents",
    )

    # Create pipeline with GPU adapters
    pipeline = AdvancedRAGPipeline(
        llm_adapter=llm,
        embedding_adapter=embedding,
        vectordb_adapter=vectordb,
    )

    print("GPU Pipeline created:")
    print(f"  - LLM: vLLM @ gpu-server:8000")
    print(f"  - Embedding: Infinity @ gpu-server:7997")
    print(f"  - VectorDB: Weaviate @ weaviate:8080")


async def switch_between_gpu_and_cpu():
    """Demonstrate switching between GPU and CPU configurations."""
    print("\n=== Switching GPU/CPU Configurations ===")

    # GPU configuration for production
    gpu_llm = VLLMAdapter(
        base_url="http://gpu-server:8000/v1",
        model="llama-3-70b",
    )

    gpu_embedding = InfinityAdapter(
        base_url="http://gpu-server:7997",
        model="BAAI/bge-m3",
        dimension=1024,
    )

    # CPU configuration for development/testing
    from rag_infra.llm import OllamaAdapter
    from rag_infra.embedding import SentenceTransformerAdapter

    cpu_llm = OllamaAdapter(
        base_url="http://localhost:11434",
        model="llama3.1:8b",
    )

    cpu_embedding = SentenceTransformerAdapter(
        model="all-MiniLM-L6-v2",
    )

    # Choose configuration based on environment
    use_gpu = os.getenv("USE_GPU", "false").lower() == "true"

    if use_gpu:
        print("Using GPU configuration:")
        pipeline = AdvancedRAGPipeline(
            llm_adapter=gpu_llm,
            embedding_adapter=gpu_embedding,
        )
    else:
        print("Using CPU configuration:")
        pipeline = AdvancedRAGPipeline(
            llm_adapter=cpu_llm,
            embedding_adapter=cpu_embedding,
        )


async def health_check_gpu_services():
    """Check health of GPU services."""
    print("\n=== GPU Services Health Check ===")

    pipeline = AdvancedRAGPipeline(
        llm_adapter=VLLMAdapter(
            base_url="http://localhost:8000/v1",
            model="test-model",
        ),
        embedding_adapter=InfinityAdapter(
            base_url="http://localhost:7997",
            model="test-model",
            dimension=1024,
        ),
    )

    try:
        health = await pipeline.health_check()
        print("Health status:")
        for service, status in health.items():
            status_str = "healthy" if status else "unavailable"
            print(f"  - {service}: {status_str}")
    except Exception as e:
        print(f"Health check failed: {e}")
        print("(This is expected if GPU services are not running)")


async def main():
    print("=== GPU Service Configuration Examples ===\n")

    await create_gpu_pipeline_from_env()
    await create_gpu_pipeline_programmatic()
    await switch_between_gpu_and_cpu()
    await health_check_gpu_services()


if __name__ == "__main__":
    asyncio.run(main())
