"""Query Rewrite Use Case."""

from dataclasses import dataclass

from rag_core.domain.entities.query import Query, RewrittenQuery
from rag_core.domain.interfaces.llm_port import LLMPort


QUERY_REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant. Your task is to improve search queries for better retrieval from a knowledge base.

Given a user query, rewrite it to:
1. Be more specific and detailed
2. Include synonyms and related terms
3. Clarify ambiguous terms
4. Expand acronyms if applicable

Respond with ONLY the rewritten query, nothing else."""


@dataclass
class QueryRewriteUseCase:
    """Use case for rewriting queries to improve retrieval.

    This use case takes a user query and rewrites it using an LLM
    to improve the chances of retrieving relevant documents.
    """

    llm: LLMPort
    system_prompt: str = QUERY_REWRITE_SYSTEM_PROMPT

    async def execute(
        self,
        query: Query,
        include_expansions: bool = True,
    ) -> RewrittenQuery:
        """Execute the query rewrite use case.

        Args:
            query: The original user query.
            include_expansions: Whether to include expansion terms.

        Returns:
            A RewrittenQuery with the improved query text.
        """
        # Generate rewritten query using LLM
        response = await self.llm.generate(
            prompt=f"Rewrite this search query for better retrieval: {query.text}",
            system_prompt=self.system_prompt,
            temperature=0.3,
            max_tokens=200,
        )

        rewritten_text = response.content.strip()

        # Extract expansion terms if requested
        expansion_terms: list[str] = []
        if include_expansions:
            expansion_terms = await self._extract_expansion_terms(query.text)

        return RewrittenQuery(
            original_query=query,
            rewritten_text=rewritten_text,
            expansion_terms=expansion_terms,
            rewrite_strategy="llm_rewrite",
            metadata={
                "model": response.model,
                "tokens_used": response.total_tokens,
            },
        )

    async def _extract_expansion_terms(self, query_text: str) -> list[str]:
        """Extract expansion terms for the query.

        Args:
            query_text: The original query text.

        Returns:
            List of expansion terms.
        """
        expansion_prompt = f"""Given the search query: "{query_text}"

List 3-5 related search terms or synonyms that could help find relevant documents.
Respond with only the terms, one per line."""

        response = await self.llm.generate(
            prompt=expansion_prompt,
            system_prompt="You are a search term expansion assistant.",
            temperature=0.3,
            max_tokens=100,
        )

        # Parse expansion terms from response
        terms = [
            term.strip().strip("-").strip("•").strip()
            for term in response.content.strip().split("\n")
            if term.strip()
        ]

        return terms[:5]  # Limit to 5 terms

    async def execute_hyde(self, query: Query) -> str:
        """Generate a hypothetical document for HyDE retrieval.

        HyDE (Hypothetical Document Embeddings) generates a hypothetical
        answer to the query, which is then used for embedding-based retrieval.

        Args:
            query: The original user query.

        Returns:
            A hypothetical document/answer text.
        """
        hyde_prompt = f"""Given the question: "{query.text}"

Write a detailed paragraph that would be a perfect answer to this question.
The answer should be factual and informative, as if it came from a knowledge base."""

        response = await self.llm.generate(
            prompt=hyde_prompt,
            system_prompt="You are a knowledgeable assistant writing informative answers.",
            temperature=0.5,
            max_tokens=300,
        )

        return response.content.strip()
