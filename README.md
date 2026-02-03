# Advanced RAG Pipeline

<p align="center">English | <a href="README.ko.md">한국어</a></p>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![UV](https://img.shields.io/badge/UV-Package%20Manager-5C4EE5)](https://github.com/astral-sh/uv)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-ready RAG Pipeline with Clean Architecture and Pluggable Adapters**

A modular, extensible Retrieval-Augmented Generation (RAG) pipeline built with Clean Architecture principles. Easily swap between different LLM providers, embedding services, and vector databases without changing your application code.

---

## Highlights

- **Clean Architecture**: Strict separation between Domain, Application, Infrastructure, and Interface layers
- **Port-Adapter Pattern**: Plug in any LLM, embedding, or vector database provider
- **GPU-Ready**: Seamless switching between CPU (OpenAI) and GPU services (vLLM, Infinity)
- **Async-First**: Built with asyncio for high-performance concurrent operations
- **Type-Safe**: Full type hints and Pydantic validation
- **REST API**: Production-ready FastAPI endpoints with streaming support

## Supported Providers

| Component | Providers |
|-----------|-----------|
| **LLM** | OpenAI, vLLM, Ollama |
| **Embedding** | OpenAI, Infinity, SentenceTransformers |
| **Vector DB** | Weaviate, ChromaDB |
| **Chunking** | Semantic, Fixed-size |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│         AdvancedRAGPipeline (Facade) + REST API             │
├─────────────────────────────────────────────────────────────┤
│                   Application Layer                         │
│     Use Cases: Query, Retrieve, Generate, Ingest            │
│     Services: PipelineService, IngestService, QueryService  │
├─────────────────────────────────────────────────────────────┤
│                     Domain Layer                            │
│     Entities: Document, Chunk, Query, Response              │
│     Ports: LLMPort, EmbeddingPort, VectorDBPort             │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                       │
│     Adapters: OpenAI, vLLM, Weaviate, ChromaDB, etc.        │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
rag-advanced-pipeline/
├── pyproject.toml                 # UV workspace configuration
├── packages/
│   ├── core/                      # Domain + Application Layer
│   │   └── src/rag_core/
│   │       ├── domain/
│   │       │   ├── entities/      # Document, Chunk, Query, Response
│   │       │   ├── value_objects/ # Embedding, Score
│   │       │   └── interfaces/    # Port definitions (LLM, Embedding, VectorDB)
│   │       └── application/
│   │           ├── use_cases/     # QueryRewrite, Retrieve, Generate, Ingest
│   │           ├── services/      # Orchestration services
│   │           └── dto/           # Data Transfer Objects
│   ├── infrastructure/            # Adapter implementations
│   │   └── src/rag_infra/
│   │       ├── llm/               # OpenAI, vLLM, Ollama adapters
│   │       ├── embedding/         # OpenAI, Infinity, SentenceTransformers
│   │       ├── vectordb/          # Weaviate, ChromaDB adapters
│   │       └── chunking/          # Semantic, Fixed chunkers
│   └── interface/                 # External interfaces
│       └── src/rag_interface/
│           ├── facade/            # AdvancedRAGPipeline
│           └── api/               # FastAPI routes
├── examples/                      # Usage examples
└── tests/                         # Unit and integration tests
```

---

## Installation

### Prerequisites

- Python 3.11+
- [UV](https://github.com/astral-sh/uv) package manager (recommended)

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/Hyeongseob91/rag-advanced-pipeline.git
cd rag-advanced-pipeline

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

uv sync

# Install with dev dependencies
uv sync --all-extras
```

### Using pip

```bash
pip install -e .
```

---

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### 2. Basic Usage

```python
import asyncio
from rag_interface import AdvancedRAGPipeline

async def main():
    # Create pipeline (auto-configures from environment)
    pipeline = AdvancedRAGPipeline()

    # Ingest a document
    result = await pipeline.ingest(
        content="Your document content here...",
        source="document.txt",
        metadata={"category": "technical"},
    )
    print(f"Ingested {result.chunk_count} chunks")

    # Query the pipeline
    response = await pipeline.query(
        question="What is this document about?",
        top_k=5,
    )
    print(f"Answer: {response.answer}")

    # Print sources
    for source in response.sources:
        print(f"  [{source.score:.2f}] {source.content[:100]}...")

asyncio.run(main())
```

### 3. Using GPU Services

```python
from rag_interface import AdvancedRAGPipeline
from rag_infra.llm import VLLMAdapter
from rag_infra.embedding import InfinityAdapter

# Configure GPU-accelerated services
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

---

## Data Schema

### Domain Entities

#### Document

```python
@dataclass
class Document:
    content: str                    # Full document text
    source: str                     # Source identifier (filename, URL)
    metadata: dict[str, Any]        # Custom metadata
    id: UUID                        # Unique identifier
    created_at: datetime            # Creation timestamp
    chunks: list[Chunk]             # Associated chunks
```

#### Chunk

```python
@dataclass
class Chunk:
    content: str                    # Chunk text
    document_id: UUID               # Parent document ID
    chunk_index: int                # Position in document
    metadata: dict[str, Any]        # Inherited + custom metadata
    id: UUID                        # Unique identifier
    embedding: list[float] | None   # Vector embedding
    start_char: int | None          # Start position in document
    end_char: int | None            # End position in document
```

#### Query

```python
@dataclass
class Query:
    text: str                       # Original query text
    metadata: dict[str, Any]        # Query metadata
    id: UUID                        # Unique identifier
    created_at: datetime            # Timestamp
```

### Request/Response DTOs

#### QueryRequest

```python
@dataclass
class QueryRequest:
    query: str                      # Question to answer
    top_k: int = 5                  # Number of chunks to retrieve
    score_threshold: float | None   # Minimum relevance score (0-1)
    rewrite_query: bool = True      # Enable query rewriting
    collection_name: str | None     # Target collection
    filter_metadata: dict | None    # Metadata filters
```

#### QueryResponse

```python
@dataclass
class QueryResponse:
    query_id: UUID                  # Request identifier
    original_query: str             # Original question
    rewritten_query: str | None     # Rewritten query (if enabled)
    answer: str                     # Generated answer
    sources: list[SourceDTO]        # Retrieved sources
    model: str                      # LLM model used
    total_tokens: int               # Token usage
    retrieval_time_ms: float        # Retrieval latency
    generation_time_ms: float       # Generation latency
    total_time_ms: float            # Total latency
```

#### IngestRequest

```python
@dataclass
class IngestRequest:
    content: str                    # Document content
    source: str                     # Source identifier
    collection_name: str | None     # Target collection
    chunk_size: int = 512           # Characters per chunk
    chunk_overlap: int = 50         # Overlap between chunks
    metadata: dict[str, Any]        # Document metadata
```

#### IngestResponse

```python
@dataclass
class IngestResponse:
    document_id: str                # Created document ID
    chunk_count: int                # Number of chunks created
    total_characters: int           # Total content length
    collection_name: str            # Target collection
    processing_time_ms: float       # Processing latency
```

---

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

#### LLM Settings

```bash
# Provider: openai, vllm, ollama
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=your-api-key
LLM_BASE_URL=                    # Optional: custom endpoint
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2000
```

#### Embedding Settings

```bash
# Provider: openai, infinity, sentence_transformers
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-api-key
EMBEDDING_DIMENSION=1536
```

#### Vector Database Settings

```bash
# Provider: weaviate, chroma
VECTORDB_PROVIDER=weaviate
VECTORDB_HOST=localhost
VECTORDB_PORT=8080
VECTORDB_COLLECTION_NAME=documents
```

#### Chunking Settings

```bash
# Type: semantic, fixed
CHUNKER_TYPE=semantic
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

#### Retrieval Settings

```bash
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.7
```

---

## REST API

### Start the Server

```bash
uvicorn rag_interface.api.routes:create_app --factory --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Execute RAG query |
| `POST` | `/api/v1/query/stream` | Streaming RAG query |
| `POST` | `/api/v1/search` | Search without generation |
| `POST` | `/api/v1/ingest` | Ingest document |
| `DELETE` | `/api/v1/documents/{id}` | Delete document |
| `GET` | `/api/v1/health` | Health check |

### API Examples

#### Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "top_k": 5,
    "rewrite_query": true
  }'
```

**Response:**

```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "original_query": "What is machine learning?",
  "rewritten_query": "machine learning definition types applications",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "content": "Machine learning enables systems to learn...",
      "score": 0.92,
      "rank": 1
    }
  ],
  "model": "gpt-4o",
  "total_tokens": 523,
  "retrieval_time_ms": 45.2,
  "generation_time_ms": 1230.5,
  "total_time_ms": 1275.7
}
```

#### Ingest Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your document content...",
    "source": "document.txt",
    "chunk_size": 512,
    "metadata": {"category": "technical"}
  }'
```

**Response:**

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440001",
  "chunk_count": 5,
  "total_characters": 2048,
  "collection_name": "documents",
  "processing_time_ms": 234.5
}
```

#### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**

```json
{
  "llm": true,
  "embedding": true,
  "vectordb": true,
  "overall": true
}
```

---

## Pipeline Methods

### AdvancedRAGPipeline API

| Method | Description |
|--------|-------------|
| `query(question, top_k, ...)` | Execute full RAG pipeline |
| `query_stream(question, ...)` | Streaming response |
| `search(query, top_k, ...)` | Search without generation |
| `ingest(content, source, ...)` | Ingest single document |
| `ingest_batch(documents, ...)` | Batch ingestion |
| `delete_document(document_id)` | Delete document and chunks |
| `get_context(query, top_k)` | Get context without generation |
| `health_check()` | Check all services |

---

## Extending the Pipeline

### Custom LLM Adapter

```python
from rag_core.domain.interfaces.llm_port import LLMPort, LLMResponse

class MyLLMAdapter(LLMPort):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Your implementation
        return LLMResponse(text="...", model="my-model")

    async def generate_stream(self, prompt: str, **kwargs):
        yield "streaming "
        yield "response"

    async def generate_with_messages(self, messages, **kwargs) -> LLMResponse:
        # Chat-style implementation
        pass

    @property
    def model_name(self) -> str:
        return "my-custom-model"

    async def health_check(self) -> bool:
        return True
```

### Custom Embedding Adapter

```python
from rag_core.domain.interfaces.embedding_port import EmbeddingPort, EmbeddingResult

class MyEmbeddingAdapter(EmbeddingPort):
    async def embed(self, text: str) -> list[float]:
        # Return embedding vector
        return [0.1, 0.2, ...]

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        # Batch embedding
        return EmbeddingResult(embeddings=[...], model="my-model")

    @property
    def model_name(self) -> str:
        return "my-embedding-model"

    @property
    def dimension(self) -> int:
        return 768

    async def health_check(self) -> bool:
        return True
```

### Custom Vector Database Adapter

```python
from rag_core.domain.interfaces.vectordb_port import VectorDBPort

class MyVectorDBAdapter(VectorDBPort):
    async def upsert(self, chunks, collection_name=None) -> int:
        # Store chunks with embeddings
        return len(chunks)

    async def search(self, embedding, top_k, collection_name=None, **kwargs):
        # Return similar chunks
        return [...]

    async def delete(self, ids, collection_name=None) -> int:
        # Delete by IDs
        return len(ids)

    async def health_check(self) -> bool:
        return True
```

---

## Development

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov

# Specific tests
uv run pytest tests/unit/test_domain/
```

### Code Quality

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy packages/
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
