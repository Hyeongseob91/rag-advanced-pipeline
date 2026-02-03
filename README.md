# Advanced RAG Pipeline

A Clean Architecture + MSA implementation of an Advanced RAG (Retrieval-Augmented Generation) Pipeline with pluggable adapters for LLM, embedding, and vector database services.

## Features

- **Clean Architecture**: Separation of concerns with Domain, Application, Infrastructure, and Interface layers
- **Port-Adapter Pattern**: Pluggable adapters for all external services
- **GPU Service Support**: Easy swapping between CPU and GPU services (vLLM, Infinity)
- **UV Workspace**: Modular package structure with workspace dependencies
- **Multiple Providers**:
  - LLM: OpenAI, vLLM, Ollama
  - Embedding: OpenAI, Infinity, SentenceTransformers
  - VectorDB: Weaviate, ChromaDB
  - Chunking: Semantic, Fixed-size

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│              (API, CLI, Pipeline Facade)                    │
├─────────────────────────────────────────────────────────────┤
│                   Application Layer                         │
│              (Use Cases, Services, DTOs)                    │
├─────────────────────────────────────────────────────────────┤
│                     Domain Layer                            │
│          (Entities, Value Objects, Interfaces)              │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                       │
│         (Adapters: LLM, Embedding, VectorDB)                │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag-advanced-pipeline/
├── pyproject.toml              # Workspace configuration
├── packages/
│   ├── core/                   # Domain + Application Layer
│   │   └── src/rag_core/
│   │       ├── domain/         # Entities, Value Objects, Ports
│   │       └── application/    # Services, Use Cases, DTOs
│   ├── infrastructure/         # Adapters for external services
│   │   └── src/rag_infra/
│   │       ├── llm/            # OpenAI, vLLM, Ollama
│   │       ├── embedding/      # OpenAI, Infinity, SentenceTransformers
│   │       ├── vectordb/       # Weaviate, ChromaDB
│   │       └── chunking/       # Semantic, Fixed chunkers
│   └── interface/              # Facade and API
│       └── src/rag_interface/
│           ├── facade/         # AdvancedRAGPipeline
│           └── api/            # REST API routes
├── examples/                   # Usage examples
└── tests/                      # Unit and integration tests
```

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
cd rag-advanced-pipeline

# Install with UV
uv sync

# Install with dev dependencies
uv sync --all-extras
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
import asyncio
from rag_interface import AdvancedRAGPipeline

async def main():
    # Create pipeline with default settings from environment
    pipeline = AdvancedRAGPipeline()

    # Ingest a document
    await pipeline.ingest(
        content="Your document content here...",
        source="document.txt",
    )

    # Query the pipeline
    result = await pipeline.query("What is this document about?")
    print(result.answer)

asyncio.run(main())
```

### Custom GPU Configuration

```python
from rag_interface import AdvancedRAGPipeline
from rag_infra.llm import VLLMAdapter
from rag_infra.embedding import InfinityAdapter

# Create pipeline with GPU services
pipeline = AdvancedRAGPipeline(
    llm_adapter=VLLMAdapter(
        base_url="http://gpu-server:8000/v1",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ),
    embedding_adapter=InfinityAdapter(
        base_url="http://gpu-server:7997",
        model="BAAI/bge-m3",
        dimension=1024,
    ),
)
```

## Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

### Environment Variables

#### LLM Configuration

```bash
LLM_PROVIDER=openai          # openai, vllm, ollama
LLM_MODEL=gpt-4o
LLM_API_KEY=your-api-key
LLM_BASE_URL=                # Optional custom base URL
```

#### Embedding Configuration

```bash
EMBEDDING_PROVIDER=openai    # openai, infinity, sentence_transformers
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-api-key
EMBEDDING_DIMENSION=1536
```

#### Vector Database Configuration

```bash
VECTORDB_PROVIDER=weaviate   # weaviate, chroma
VECTORDB_HOST=localhost
VECTORDB_PORT=8080
VECTORDB_COLLECTION_NAME=documents
```

#### Chunking Configuration

```bash
CHUNKER_TYPE=semantic        # semantic, fixed
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## API Usage

### Start the API Server

```bash
uvicorn rag_interface.api.routes:create_app --factory --reload
```

### API Endpoints

- `POST /api/v1/query` - Execute a RAG query
- `POST /api/v1/query/stream` - Execute with streaming response
- `POST /api/v1/search` - Search without generation
- `POST /api/v1/ingest` - Ingest a document
- `DELETE /api/v1/documents/{id}` - Delete a document
- `GET /api/v1/health` - Health check

### Example API Request

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 5}'
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/unit/test_domain/test_entities.py
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy packages/
```

## Extending the Pipeline

### Custom LLM Adapter

```python
from rag_core.domain.interfaces.llm_port import LLMPort, LLMResponse

class MyCustomLLMAdapter(LLMPort):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Your implementation
        pass

    async def generate_stream(self, prompt: str, **kwargs):
        # Your streaming implementation
        pass

    async def generate_with_messages(self, messages, **kwargs) -> LLMResponse:
        # Your implementation
        pass

    @property
    def model_name(self) -> str:
        return "my-custom-model"
```

### Custom Embedding Adapter

```python
from rag_core.domain.interfaces.embedding_port import EmbeddingPort, EmbeddingResult

class MyCustomEmbeddingAdapter(EmbeddingPort):
    async def embed(self, text: str) -> list[float]:
        # Your implementation
        pass

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        # Your implementation
        pass

    @property
    def model_name(self) -> str:
        return "my-custom-embedding"

    @property
    def dimension(self) -> int:
        return 768
```

## License

MIT License
