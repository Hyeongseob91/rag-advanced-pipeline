# Advanced RAG Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![UV](https://img.shields.io/badge/UV-Package%20Manager-5C4EE5)](https://github.com/astral-sh/uv)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)

> **Clean Architecture와 플러그형 어댑터를 적용한 프로덕션 수준의 RAG 파이프라인**

Clean Architecture 원칙을 기반으로 구축된 모듈형, 확장 가능한 RAG(Retrieval-Augmented Generation) 파이프라인입니다. 애플리케이션 코드 변경 없이 다양한 LLM, 임베딩, 벡터 데이터베이스 제공자를 쉽게 교체할 수 있습니다.

---

## 주요 특징

- **Clean Architecture**: Domain, Application, Infrastructure, Interface 계층 간 엄격한 분리
- **Port-Adapter 패턴**: LLM, 임베딩, 벡터 DB 제공자를 자유롭게 교체 가능
- **GPU 지원**: CPU(OpenAI)와 GPU 서비스(vLLM, Infinity) 간 원활한 전환
- **비동기 우선**: asyncio 기반 고성능 동시 처리
- **타입 안전성**: 완벽한 타입 힌트와 Pydantic 검증
- **REST API**: 스트리밍 지원을 포함한 프로덕션 수준의 FastAPI 엔드포인트

## 지원 제공자

| 컴포넌트 | 제공자 |
|----------|--------|
| **LLM** | OpenAI, vLLM, Ollama |
| **임베딩** | OpenAI, Infinity, SentenceTransformers |
| **벡터 DB** | Weaviate, ChromaDB |
| **청킹** | Semantic, Fixed-size |

---

## 아키텍처

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
│     Adapters: OpenAI, vLLM, Weaviate, ChromaDB 등           │
└─────────────────────────────────────────────────────────────┘
```

### 프로젝트 구조

```
rag-advanced-pipeline/
├── pyproject.toml                 # UV 워크스페이스 설정
├── packages/
│   ├── core/                      # Domain + Application 계층
│   │   └── src/rag_core/
│   │       ├── domain/
│   │       │   ├── entities/      # Document, Chunk, Query, Response
│   │       │   ├── value_objects/ # Embedding, Score
│   │       │   └── interfaces/    # 포트 정의 (LLM, Embedding, VectorDB)
│   │       └── application/
│   │           ├── use_cases/     # QueryRewrite, Retrieve, Generate, Ingest
│   │           ├── services/      # 오케스트레이션 서비스
│   │           └── dto/           # 데이터 전송 객체
│   ├── infrastructure/            # 어댑터 구현체
│   │   └── src/rag_infra/
│   │       ├── llm/               # OpenAI, vLLM, Ollama 어댑터
│   │       ├── embedding/         # OpenAI, Infinity, SentenceTransformers
│   │       ├── vectordb/          # Weaviate, ChromaDB 어댑터
│   │       └── chunking/          # Semantic, Fixed 청커
│   └── interface/                 # 외부 인터페이스
│       └── src/rag_interface/
│           ├── facade/            # AdvancedRAGPipeline
│           └── api/               # FastAPI 라우트
├── examples/                      # 사용 예제
└── tests/                         # 단위 및 통합 테스트
```

---

## 설치

### 사전 요구사항

- Python 3.11+
- [UV](https://github.com/astral-sh/uv) 패키지 매니저 (권장)

### UV 사용 (권장)

```bash
# 저장소 복제
git clone https://github.com/Hyeongseob91/rag-advanced-pipeline.git
cd rag-advanced-pipeline

# 가상 환경 생성 및 설치
uv venv
source .venv/bin/activate  # Linux/macOS
# 또는 .venv\Scripts\activate  # Windows

uv sync

# 개발 의존성 포함 설치
uv sync --all-extras
```

### pip 사용

```bash
pip install -e .
```

---

## 빠른 시작

### 1. 환경 설정

```bash
cp .env.example .env
# .env 파일에서 API 키와 설정 수정
```

### 2. 기본 사용법

```python
import asyncio
from rag_interface import AdvancedRAGPipeline

async def main():
    # 파이프라인 생성 (환경 변수에서 자동 설정)
    pipeline = AdvancedRAGPipeline()

    # 문서 수집
    result = await pipeline.ingest(
        content="문서 내용을 여기에 입력...",
        source="document.txt",
        metadata={"category": "technical"},
    )
    print(f"{result.chunk_count}개의 청크 생성됨")

    # 파이프라인 쿼리
    response = await pipeline.query(
        question="이 문서는 무엇에 관한 내용인가요?",
        top_k=5,
    )
    print(f"답변: {response.answer}")

    # 출처 출력
    for source in response.sources:
        print(f"  [{source.score:.2f}] {source.content[:100]}...")

asyncio.run(main())
```

### 3. GPU 서비스 사용

```python
from rag_interface import AdvancedRAGPipeline
from rag_infra.llm import VLLMAdapter
from rag_infra.embedding import InfinityAdapter

# GPU 가속 서비스 설정
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

## 데이터 스키마

### 도메인 엔티티

#### Document (문서)

```python
@dataclass
class Document:
    content: str                    # 전체 문서 텍스트
    source: str                     # 출처 식별자 (파일명, URL)
    metadata: dict[str, Any]        # 커스텀 메타데이터
    id: UUID                        # 고유 식별자
    created_at: datetime            # 생성 시간
    chunks: list[Chunk]             # 연관된 청크들
```

#### Chunk (청크)

```python
@dataclass
class Chunk:
    content: str                    # 청크 텍스트
    document_id: UUID               # 부모 문서 ID
    chunk_index: int                # 문서 내 위치
    metadata: dict[str, Any]        # 상속 + 커스텀 메타데이터
    id: UUID                        # 고유 식별자
    embedding: list[float] | None   # 벡터 임베딩
    start_char: int | None          # 문서 내 시작 위치
    end_char: int | None            # 문서 내 끝 위치
```

#### Query (쿼리)

```python
@dataclass
class Query:
    text: str                       # 원본 쿼리 텍스트
    metadata: dict[str, Any]        # 쿼리 메타데이터
    id: UUID                        # 고유 식별자
    created_at: datetime            # 타임스탬프
```

### Request/Response DTO

#### QueryRequest (쿼리 요청)

```python
@dataclass
class QueryRequest:
    query: str                      # 답변할 질문
    top_k: int = 5                  # 검색할 청크 수
    score_threshold: float | None   # 최소 관련성 점수 (0-1)
    rewrite_query: bool = True      # 쿼리 재작성 활성화
    collection_name: str | None     # 대상 컬렉션
    filter_metadata: dict | None    # 메타데이터 필터
```

#### QueryResponse (쿼리 응답)

```python
@dataclass
class QueryResponse:
    query_id: UUID                  # 요청 식별자
    original_query: str             # 원본 질문
    rewritten_query: str | None     # 재작성된 쿼리 (활성화 시)
    answer: str                     # 생성된 답변
    sources: list[SourceDTO]        # 검색된 출처
    model: str                      # 사용된 LLM 모델
    total_tokens: int               # 토큰 사용량
    retrieval_time_ms: float        # 검색 지연시간
    generation_time_ms: float       # 생성 지연시간
    total_time_ms: float            # 총 지연시간
```

#### IngestRequest (수집 요청)

```python
@dataclass
class IngestRequest:
    content: str                    # 문서 내용
    source: str                     # 출처 식별자
    collection_name: str | None     # 대상 컬렉션
    chunk_size: int = 512           # 청크당 문자 수
    chunk_overlap: int = 50         # 청크 간 중첩
    metadata: dict[str, Any]        # 문서 메타데이터
```

#### IngestResponse (수집 응답)

```python
@dataclass
class IngestResponse:
    document_id: str                # 생성된 문서 ID
    chunk_count: int                # 생성된 청크 수
    total_characters: int           # 총 콘텐츠 길이
    collection_name: str            # 대상 컬렉션
    processing_time_ms: float       # 처리 지연시간
```

---

## 설정

### 환경 변수

템플릿에서 `.env` 파일 생성:

```bash
cp .env.example .env
```

#### LLM 설정

```bash
# Provider: openai, vllm, ollama
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=your-api-key
LLM_BASE_URL=                    # 선택: 커스텀 엔드포인트
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2000
```

#### 임베딩 설정

```bash
# Provider: openai, infinity, sentence_transformers
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your-api-key
EMBEDDING_DIMENSION=1536
```

#### 벡터 데이터베이스 설정

```bash
# Provider: weaviate, chroma
VECTORDB_PROVIDER=weaviate
VECTORDB_HOST=localhost
VECTORDB_PORT=8080
VECTORDB_COLLECTION_NAME=documents
```

#### 청킹 설정

```bash
# Type: semantic, fixed
CHUNKER_TYPE=semantic
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

#### 검색 설정

```bash
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.7
```

---

## REST API

### 서버 시작

```bash
uvicorn rag_interface.api.routes:create_app --factory --host 0.0.0.0 --port 8000
```

### 엔드포인트

| 메서드 | 엔드포인트 | 설명 |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | RAG 쿼리 실행 |
| `POST` | `/api/v1/query/stream` | 스트리밍 RAG 쿼리 |
| `POST` | `/api/v1/search` | 생성 없이 검색만 |
| `POST` | `/api/v1/ingest` | 문서 수집 |
| `DELETE` | `/api/v1/documents/{id}` | 문서 삭제 |
| `GET` | `/api/v1/health` | 헬스 체크 |

### API 예제

#### 쿼리

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "머신러닝이란 무엇인가요?",
    "top_k": 5,
    "rewrite_query": true
  }'
```

**응답:**

```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "original_query": "머신러닝이란 무엇인가요?",
  "rewritten_query": "머신러닝 정의 종류 응용분야",
  "answer": "머신러닝은 인공지능의 하위 분야로...",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "content": "머신러닝은 시스템이 학습할 수 있도록...",
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

#### 문서 수집

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "문서 내용...",
    "source": "document.txt",
    "chunk_size": 512,
    "metadata": {"category": "technical"}
  }'
```

**응답:**

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440001",
  "chunk_count": 5,
  "total_characters": 2048,
  "collection_name": "documents",
  "processing_time_ms": 234.5
}
```

#### 헬스 체크

```bash
curl http://localhost:8000/api/v1/health
```

**응답:**

```json
{
  "llm": true,
  "embedding": true,
  "vectordb": true,
  "overall": true
}
```

---

## 파이프라인 메서드

### AdvancedRAGPipeline API

| 메서드 | 설명 |
|--------|-------------|
| `query(question, top_k, ...)` | 전체 RAG 파이프라인 실행 |
| `query_stream(question, ...)` | 스트리밍 응답 |
| `search(query, top_k, ...)` | 생성 없이 검색만 |
| `ingest(content, source, ...)` | 단일 문서 수집 |
| `ingest_batch(documents, ...)` | 배치 수집 |
| `delete_document(document_id)` | 문서 및 청크 삭제 |
| `get_context(query, top_k)` | 생성 없이 컨텍스트 가져오기 |
| `health_check()` | 모든 서비스 상태 확인 |

---

## 파이프라인 확장

### 커스텀 LLM 어댑터

```python
from rag_core.domain.interfaces.llm_port import LLMPort, LLMResponse

class MyLLMAdapter(LLMPort):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # 구현
        return LLMResponse(text="...", model="my-model")

    async def generate_stream(self, prompt: str, **kwargs):
        yield "스트리밍 "
        yield "응답"

    async def generate_with_messages(self, messages, **kwargs) -> LLMResponse:
        # 채팅 스타일 구현
        pass

    @property
    def model_name(self) -> str:
        return "my-custom-model"

    async def health_check(self) -> bool:
        return True
```

### 커스텀 임베딩 어댑터

```python
from rag_core.domain.interfaces.embedding_port import EmbeddingPort, EmbeddingResult

class MyEmbeddingAdapter(EmbeddingPort):
    async def embed(self, text: str) -> list[float]:
        # 임베딩 벡터 반환
        return [0.1, 0.2, ...]

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        # 배치 임베딩
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

### 커스텀 벡터 데이터베이스 어댑터

```python
from rag_core.domain.interfaces.vectordb_port import VectorDBPort

class MyVectorDBAdapter(VectorDBPort):
    async def upsert(self, chunks, collection_name=None) -> int:
        # 임베딩과 함께 청크 저장
        return len(chunks)

    async def search(self, embedding, top_k, collection_name=None, **kwargs):
        # 유사한 청크 반환
        return [...]

    async def delete(self, ids, collection_name=None) -> int:
        # ID로 삭제
        return len(ids)

    async def health_check(self) -> bool:
        return True
```

---

## 개발

### 테스트 실행

```bash
# 전체 테스트
uv run pytest

# 커버리지 포함
uv run pytest --cov

# 특정 테스트
uv run pytest tests/unit/test_domain/
```

### 코드 품질

```bash
# 포맷팅
uv run ruff format .

# 린트
uv run ruff check .

# 타입 체크
uv run mypy packages/
```

---

## 라이선스

MIT 라이선스 - 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

---

## 기여

기여를 환영합니다! Pull Request를 자유롭게 제출해 주세요.

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 열기
