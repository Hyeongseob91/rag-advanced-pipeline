"""REST API routes for the RAG pipeline."""

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_interface.facade.pipeline_facade import AdvancedRAGPipeline


# Request/Response models
class QueryRequestModel(BaseModel):
    """Request model for RAG query."""

    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    rewrite_query: bool = Field(default=True, description="Whether to rewrite the query")
    collection_name: str | None = Field(default=None, description="Collection to search")
    score_threshold: float | None = Field(
        default=None, ge=0, le=1, description="Minimum relevance score"
    )
    filter_metadata: dict[str, Any] | None = Field(
        default=None, description="Metadata filters"
    )


class SourceModel(BaseModel):
    """Model for a source/citation."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponseModel(BaseModel):
    """Response model for RAG query."""

    query_id: str
    original_query: str
    rewritten_query: str | None
    answer: str
    sources: list[SourceModel]
    model: str
    total_tokens: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0


class IngestRequestModel(BaseModel):
    """Request model for document ingestion."""

    content: str = Field(..., min_length=1, description="Document content")
    source: str = Field(..., min_length=1, description="Source identifier")
    collection_name: str | None = Field(default=None, description="Target collection")
    chunk_size: int = Field(default=512, ge=100, le=4000, description="Chunk size")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Chunk overlap")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class IngestResponseModel(BaseModel):
    """Response model for document ingestion."""

    document_id: str
    chunk_count: int
    total_characters: int
    collection_name: str
    processing_time_ms: float = 0.0


class SearchRequestModel(BaseModel):
    """Request model for search."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    collection_name: str | None = Field(default=None, description="Collection to search")
    score_threshold: float | None = Field(default=None, ge=0, le=1)


class HealthResponseModel(BaseModel):
    """Response model for health check."""

    llm: bool
    embedding: bool
    vectordb: bool
    overall: bool


# Router setup
router = APIRouter(prefix="/api/v1", tags=["RAG"])

# Pipeline instance (created lazily)
_pipeline: AdvancedRAGPipeline | None = None


def get_pipeline() -> AdvancedRAGPipeline:
    """Get or create the pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AdvancedRAGPipeline()
    return _pipeline


def set_pipeline(pipeline: AdvancedRAGPipeline) -> None:
    """Set a custom pipeline instance."""
    global _pipeline
    _pipeline = pipeline


@router.post("/query", response_model=QueryResponseModel)
async def query(request: QueryRequestModel) -> QueryResponseModel:
    """Execute a RAG query and return the generated answer with sources."""
    try:
        pipeline = get_pipeline()
        result = await pipeline.query(
            question=request.question,
            top_k=request.top_k,
            rewrite_query=request.rewrite_query,
            collection_name=request.collection_name,
            score_threshold=request.score_threshold,
            filter_metadata=request.filter_metadata,
        )

        return QueryResponseModel(
            query_id=str(result.query_id),
            original_query=result.original_query,
            rewritten_query=result.rewritten_query,
            answer=result.answer,
            sources=[
                SourceModel(
                    chunk_id=str(s.chunk_id),
                    document_id=str(s.document_id),
                    content=s.content,
                    score=s.score,
                    rank=s.rank,
                    metadata=s.metadata,
                )
                for s in result.sources
            ],
            model=result.model,
            total_tokens=result.total_tokens,
            retrieval_time_ms=result.retrieval_time_ms,
            generation_time_ms=result.generation_time_ms,
            total_time_ms=result.total_time_ms,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query/stream")
async def query_stream(request: QueryRequestModel) -> StreamingResponse:
    """Execute a RAG query with streaming response."""

    async def generate():
        pipeline = get_pipeline()
        async for chunk in pipeline.query_stream(
            question=request.question,
            top_k=request.top_k,
            rewrite_query=request.rewrite_query,
            collection_name=request.collection_name,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/search", response_model=list[SourceModel])
async def search(request: SearchRequestModel) -> list[SourceModel]:
    """Search for relevant documents without generating an answer."""
    try:
        pipeline = get_pipeline()
        results = await pipeline.search(
            query=request.query,
            top_k=request.top_k,
            collection_name=request.collection_name,
            score_threshold=request.score_threshold,
        )

        return [
            SourceModel(
                chunk_id=str(s.chunk_id),
                document_id=str(s.document_id),
                content=s.content,
                score=s.score,
                rank=s.rank,
                metadata=s.metadata,
            )
            for s in results
        ]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/ingest", response_model=IngestResponseModel)
async def ingest(request: IngestRequestModel) -> IngestResponseModel:
    """Ingest a document into the RAG pipeline."""
    try:
        pipeline = get_pipeline()
        result = await pipeline.ingest(
            content=request.content,
            source=request.source,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            metadata=request.metadata,
        )

        return IngestResponseModel(
            document_id=result.document_id,
            chunk_count=result.chunk_count,
            total_characters=result.total_characters,
            collection_name=result.collection_name,
            processing_time_ms=result.processing_time_ms,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Delete a document and its chunks."""
    try:
        pipeline = get_pipeline()
        deleted_count = await pipeline.delete_document(
            document_id=document_id,
            collection_name=collection_name,
        )
        return {
            "document_id": document_id,
            "deleted_chunks": deleted_count,
            "success": True,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.get("/health", response_model=HealthResponseModel)
async def health_check() -> HealthResponseModel:
    """Check the health of all services."""
    try:
        pipeline = get_pipeline()
        status = await pipeline.health_check()
        return HealthResponseModel(
            llm=status.get("llm", False),
            embedding=status.get("embedding", False),
            vectordb=status.get("vectordb", False),
            overall=all(status.values()),
        )
    except Exception:
        return HealthResponseModel(
            llm=False,
            embedding=False,
            vectordb=False,
            overall=False,
        )


def create_app(
    title: str = "Advanced RAG Pipeline API",
    description: str = "REST API for the Advanced RAG Pipeline",
    version: str = "0.1.0",
    pipeline: AdvancedRAGPipeline | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title.
        description: API description.
        version: API version.
        pipeline: Optional custom pipeline instance.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
    )

    if pipeline:
        set_pipeline(pipeline)

    app.include_router(router)

    @app.get("/")
    async def root():
        return {"message": "Advanced RAG Pipeline API", "docs": "/docs"}

    return app
