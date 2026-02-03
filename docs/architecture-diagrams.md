# Architecture Diagrams

Mermaid 차트를 이용한 아키텍처 다이어그램 모음입니다.

---

## Recommended: C4 + RAG Pipeline Flow

Clean Architecture 계층 구조와 RAG 파이프라인 플로우를 함께 표현한 다이어그램입니다.

```mermaid
flowchart TB
    User["👤 User / Client"]

    subgraph System["Advanced RAG Pipeline"]

        subgraph Interface["🔌 Interface Layer"]
            API["REST API<br/>(FastAPI)"]
            Facade["AdvancedRAGPipeline<br/>(Facade)"]
        end

        subgraph Application["⚙️ Application Layer"]
            subgraph RAGFlow["RAG Pipeline Flow"]
                direction LR
                QR["1️⃣ QueryRewrite"]
                RT["2️⃣ Retrieve"]
                GN["3️⃣ Generate"]
            end
            subgraph IngestFlow["Ingest Flow"]
                direction LR
                CH["1️⃣ Chunk"]
                EM["2️⃣ Embed"]
                ST["3️⃣ Store"]
            end
            PipelineSvc["PipelineService"]
            IngestSvc["IngestService"]
        end

        subgraph Domain["🏛️ Domain Layer"]
            subgraph Entities["Entities"]
                Doc["Document"]
                Chunk["Chunk"]
                Query["Query"]
                Response["Response"]
            end
            subgraph Ports["Ports (Interfaces)"]
                LLMPort["LLMPort"]
                EmbPort["EmbeddingPort"]
                VDBPort["VectorDBPort"]
                ChunkPort["ChunkerPort"]
            end
        end

        subgraph Infrastructure["🔧 Infrastructure Layer"]
            subgraph LLMAdapters["LLM Adapters"]
                OpenAI_LLM["OpenAI"]
                vLLM["vLLM<br/>(GPU)"]
                Ollama["Ollama<br/>(Local)"]
            end
            subgraph EmbAdapters["Embedding Adapters"]
                OpenAI_Emb["OpenAI"]
                Infinity["Infinity<br/>(GPU)"]
                SentenceTF["SentenceTransformers<br/>(Local)"]
            end
            subgraph VDBAdapters["VectorDB Adapters"]
                Weaviate["Weaviate"]
                ChromaDB["ChromaDB"]
            end
            subgraph ChunkAdapters["Chunker"]
                Semantic["SemanticChunker"]
                Fixed["FixedChunker"]
            end
        end
    end

    subgraph External["☁️ External Services"]
        OpenAI_API["OpenAI API"]
        vLLM_Server["vLLM Server"]
        Ollama_Server["Ollama Server"]
        Infinity_Server["Infinity Server"]
        Weaviate_Server["Weaviate DB"]
        Chroma_Server["ChromaDB"]
    end

    %% User Flow
    User -->|"Query"| API
    API --> Facade
    Facade --> PipelineSvc
    Facade --> IngestSvc

    %% RAG Pipeline Flow
    PipelineSvc --> QR
    QR --> RT
    RT --> GN
    GN -->|"Answer + Sources"| Facade

    %% Ingest Flow
    IngestSvc --> CH
    CH --> EM
    EM --> ST

    %% Port Connections
    QR -.-> LLMPort
    RT -.-> EmbPort
    RT -.-> VDBPort
    GN -.-> LLMPort
    CH -.-> ChunkPort
    EM -.-> EmbPort
    ST -.-> VDBPort

    %% Adapter to External
    LLMPort --> LLMAdapters
    EmbPort --> EmbAdapters
    VDBPort --> VDBAdapters
    ChunkPort --> ChunkAdapters

    OpenAI_LLM -.->|"API"| OpenAI_API
    vLLM -.->|"API"| vLLM_Server
    Ollama -.->|"API"| Ollama_Server
    OpenAI_Emb -.->|"API"| OpenAI_API
    Infinity -.->|"API"| Infinity_Server
    Weaviate -.->|"gRPC"| Weaviate_Server
    ChromaDB -.->|"HTTP"| Chroma_Server

    %% Styling
    style System fill:#fafafa,stroke:#333,stroke-width:2px
    style Interface fill:#e3f2fd,stroke:#1565c0
    style Application fill:#f3e5f5,stroke:#7b1fa2
    style Domain fill:#fff8e1,stroke:#f57f17
    style Infrastructure fill:#e8f5e9,stroke:#2e7d32
    style External fill:#fce4ec,stroke:#c2185b
    style RAGFlow fill:#e8eaf6,stroke:#3f51b5
    style IngestFlow fill:#fce4ec,stroke:#e91e63
```

---

## Simplified: RAG Pipeline Core Flow

RAG 파이프라인의 핵심 데이터 흐름만 표현한 간소화 버전입니다.

```mermaid
flowchart LR
    subgraph Input["📥 Input"]
        Q[/"Question"/]
    end

    subgraph RAG["🔄 RAG Pipeline"]
        direction TB

        subgraph Rewrite["Step 1: Query Rewrite"]
            QR["LLM expands<br/>& clarifies query"]
        end

        subgraph Embed["Step 2: Embed & Retrieve"]
            EMB["Convert to<br/>vector embedding"]
            RET["Search similar<br/>chunks in VectorDB"]
        end

        subgraph Generate["Step 3: Generate"]
            CTX["Build context<br/>from retrieved chunks"]
            GEN["LLM generates<br/>answer with sources"]
        end
    end

    subgraph Output["📤 Output"]
        A[/"Answer +<br/>Source Citations"/]
    end

    Q --> QR
    QR --> EMB
    EMB --> RET
    RET --> CTX
    CTX --> GEN
    GEN --> A

    style Input fill:#e3f2fd,stroke:#1565c0
    style RAG fill:#f5f5f5,stroke:#333
    style Output fill:#c8e6c9,stroke:#2e7d32
    style Rewrite fill:#ffcdd2,stroke:#c62828
    style Embed fill:#fff9c4,stroke:#f9a825
    style Generate fill:#c5cae9,stroke:#303f9f
```

---

## Document Ingestion Flow

문서 수집(Ingest) 파이프라인 플로우입니다.

```mermaid
flowchart LR
    subgraph Input["📄 Input"]
        D[/"Document<br/>(text, pdf, md)"/]
    end

    subgraph Ingest["📥 Ingest Pipeline"]
        direction TB

        subgraph Chunking["Step 1: Chunking"]
            CH["Split document<br/>into chunks"]
        end

        subgraph Embedding["Step 2: Embedding"]
            EMB["Generate vector<br/>for each chunk"]
        end

        subgraph Storage["Step 3: Storage"]
            VDB["Store chunks +<br/>vectors in VectorDB"]
        end
    end

    subgraph Output["✅ Result"]
        R[/"document_id<br/>chunk_count"/]
    end

    D --> CH
    CH --> EMB
    EMB --> VDB
    VDB --> R

    style Input fill:#fff3e0,stroke:#e65100
    style Ingest fill:#f5f5f5,stroke:#333
    style Output fill:#c8e6c9,stroke:#2e7d32
    style Chunking fill:#ffccbc,stroke:#bf360c
    style Embedding fill:#b2dfdb,stroke:#00695c
    style Storage fill:#d1c4e9,stroke:#512da8
```

---

## Option 1: 계층형 블록 다이어그램

```mermaid
flowchart TB
    subgraph Interface["🔌 Interface Layer"]
        direction LR
        Facade["AdvancedRAGPipeline<br/>(Facade)"]
        API["REST API<br/>(FastAPI)"]
    end

    subgraph Application["⚙️ Application Layer"]
        direction LR
        UC["Use Cases<br/>Query | Retrieve | Generate | Ingest"]
        Services["Services<br/>PipelineService | IngestService | QueryService"]
    end

    subgraph Domain["🏛️ Domain Layer"]
        direction LR
        Entities["Entities<br/>Document | Chunk | Query | Response"]
        Ports["Ports<br/>LLMPort | EmbeddingPort | VectorDBPort"]
    end

    subgraph Infrastructure["🔧 Infrastructure Layer"]
        direction LR
        LLM["LLM Adapters<br/>OpenAI | vLLM | Ollama"]
        Embedding["Embedding Adapters<br/>OpenAI | Infinity | SentenceTransformers"]
        VectorDB["VectorDB Adapters<br/>Weaviate | ChromaDB"]
    end

    Interface --> Application
    Application --> Domain
    Domain --> Infrastructure

    style Interface fill:#e1f5fe,stroke:#01579b
    style Application fill:#f3e5f5,stroke:#4a148c
    style Domain fill:#fff3e0,stroke:#e65100
    style Infrastructure fill:#e8f5e9,stroke:#1b5e20
```

---

## Option 2: 상세 플로우 다이어그램

```mermaid
flowchart TB
    subgraph Interface["Interface Layer"]
        Facade["AdvancedRAGPipeline"]
        API["REST API"]
    end

    subgraph Application["Application Layer"]
        subgraph UseCases["Use Cases"]
            QueryRewrite["QueryRewrite"]
            Retrieve["RetrieveDocuments"]
            Generate["GenerateAnswer"]
            Ingest["IngestDocument"]
        end
        subgraph AppServices["Services"]
            Pipeline["PipelineService"]
            IngestSvc["IngestService"]
            QuerySvc["QueryService"]
        end
    end

    subgraph Domain["Domain Layer"]
        subgraph DomainEntities["Entities"]
            Doc["Document"]
            Chunk["Chunk"]
            Query["Query"]
            Response["Response"]
        end
        subgraph DomainPorts["Ports (Interfaces)"]
            LLMPort["LLMPort"]
            EmbPort["EmbeddingPort"]
            VDBPort["VectorDBPort"]
            ChunkPort["ChunkerPort"]
        end
    end

    subgraph Infra["Infrastructure Layer"]
        subgraph LLMAdapters["LLM"]
            OpenAI_LLM["OpenAI"]
            vLLM["vLLM"]
            Ollama["Ollama"]
        end
        subgraph EmbAdapters["Embedding"]
            OpenAI_Emb["OpenAI"]
            Infinity["Infinity"]
            ST["SentenceTransformers"]
        end
        subgraph VDBAdapters["VectorDB"]
            Weaviate["Weaviate"]
            Chroma["ChromaDB"]
        end
    end

    Facade --> Pipeline
    API --> Facade
    Pipeline --> QueryRewrite
    Pipeline --> Retrieve
    Pipeline --> Generate
    IngestSvc --> Ingest

    QueryRewrite --> LLMPort
    Retrieve --> EmbPort
    Retrieve --> VDBPort
    Generate --> LLMPort
    Ingest --> ChunkPort
    Ingest --> EmbPort
    Ingest --> VDBPort

    LLMPort -.-> LLMAdapters
    EmbPort -.-> EmbAdapters
    VDBPort -.-> VDBAdapters

    style Interface fill:#e3f2fd
    style Application fill:#f3e5f5
    style Domain fill:#fff8e1
    style Infra fill:#e8f5e9
```

---

## Option 3: 심플 버전

```mermaid
graph TB
    subgraph IL["Interface Layer"]
        F[AdvancedRAGPipeline + REST API]
    end

    subgraph AL["Application Layer"]
        S[Services & Use Cases]
    end

    subgraph DL["Domain Layer"]
        E[Entities & Ports]
    end

    subgraph INF["Infrastructure Layer"]
        A[Adapters: OpenAI, vLLM, Weaviate, ChromaDB]
    end

    IL --> AL --> DL --> INF

    style IL fill:#4FC3F7,color:#000
    style AL fill:#BA68C8,color:#fff
    style DL fill:#FFB74D,color:#000
    style INF fill:#81C784,color:#000
```

---

## Option 4: RAG 파이프라인 플로우

```mermaid
flowchart LR
    subgraph Input
        Q[/"Query"/]
    end

    subgraph Pipeline["RAG Pipeline"]
        direction TB
        QR["1. Query Rewrite<br/>(LLM)"]
        EMB["2. Embedding<br/>(OpenAI/Infinity)"]
        RET["3. Retrieval<br/>(Weaviate/ChromaDB)"]
        GEN["4. Generation<br/>(LLM)"]
    end

    subgraph Output
        A[/"Answer + Sources"/]
    end

    Q --> QR --> EMB --> RET --> GEN --> A

    style QR fill:#ffcdd2
    style EMB fill:#c8e6c9
    style RET fill:#bbdefb
    style GEN fill:#fff9c4
```

---

## Option 5: C4 스타일 컨테이너 다이어그램

```mermaid
flowchart TB
    User["👤 User"]

    subgraph System["Advanced RAG Pipeline"]
        subgraph Interface["Interface"]
            API["REST API<br/>FastAPI"]
            Facade["Pipeline Facade<br/>Python"]
        end

        subgraph Core["Core"]
            App["Application Services<br/>Use Cases & DTOs"]
            Domain["Domain Layer<br/>Entities & Ports"]
        end

        subgraph Adapters["Infrastructure Adapters"]
            LLM["LLM Adapter"]
            Emb["Embedding Adapter"]
            VDB["VectorDB Adapter"]
        end
    end

    subgraph External["External Services"]
        OpenAI["OpenAI API"]
        vLLM["vLLM Server"]
        Weaviate["Weaviate"]
        ChromaDB["ChromaDB"]
    end

    User -->|HTTP| API
    API --> Facade
    Facade --> App
    App --> Domain
    Domain --> Adapters

    LLM -->|API Call| OpenAI
    LLM -->|API Call| vLLM
    VDB -->|gRPC| Weaviate
    VDB -->|HTTP| ChromaDB

    style System fill:#f5f5f5,stroke:#333
    style Interface fill:#e3f2fd
    style Core fill:#fff8e1
    style Adapters fill:#e8f5e9
    style External fill:#fce4ec
```

---

## 사용 방법

1. 원하는 다이어그램을 선택합니다
2. [Mermaid Live Editor](https://mermaid.live/)에서 코드를 붙여넣기합니다
3. PNG/SVG로 내보내기합니다
4. `docs/images/` 폴더에 저장 후 README에서 참조합니다

```markdown
![Architecture](docs/images/architecture.png)
```
