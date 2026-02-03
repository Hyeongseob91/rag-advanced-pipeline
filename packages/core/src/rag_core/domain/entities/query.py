"""Query entities for RAG pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class Query:
    """An original query from the user."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")

    @property
    def cleaned_text(self) -> str:
        """Return the query text with leading/trailing whitespace removed."""
        return self.text.strip()


@dataclass
class RewrittenQuery:
    """A rewritten/expanded query for better retrieval."""

    original_query: Query
    rewritten_text: str
    expansion_terms: list[str] = field(default_factory=list)
    rewrite_strategy: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.rewritten_text or not self.rewritten_text.strip():
            raise ValueError("Rewritten query text cannot be empty")

    @property
    def original_text(self) -> str:
        """Return the original query text."""
        return self.original_query.text

    @property
    def all_search_terms(self) -> list[str]:
        """Return all search terms including original, rewritten, and expansions."""
        terms = [self.original_query.cleaned_text, self.rewritten_text.strip()]
        terms.extend(self.expansion_terms)
        return list(set(terms))


@dataclass
class HyDEQuery:
    """A Hypothetical Document Embedding query for advanced retrieval."""

    original_query: Query
    hypothetical_document: str
    embedding: list[float] | None = None
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.hypothetical_document or not self.hypothetical_document.strip():
            raise ValueError("Hypothetical document cannot be empty")
