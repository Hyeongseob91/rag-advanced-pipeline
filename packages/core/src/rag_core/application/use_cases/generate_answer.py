"""Generate Answer Use Case."""

from dataclasses import dataclass
from time import perf_counter

from rag_core.domain.entities.query import Query
from rag_core.domain.entities.response import GeneratedResponse, RetrievalResult
from rag_core.domain.interfaces.llm_port import LLMPort


DEFAULT_RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Answer the question using ONLY the information from the context provided
2. If the context doesn't contain enough information to answer, say so clearly
3. Be concise and accurate in your response
4. If you quote from the context, indicate the source

Context:
{context}"""

DEFAULT_RAG_USER_PROMPT = """Based on the context above, please answer the following question:

Question: {question}

Answer:"""


@dataclass
class GenerateAnswerUseCase:
    """Use case for generating answers using retrieved context.

    This use case takes a query and retrieved documents, then generates
    an answer using an LLM with the context.
    """

    llm: LLMPort
    system_prompt_template: str = DEFAULT_RAG_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_RAG_USER_PROMPT

    async def execute(
        self,
        query: Query,
        context: str,
        sources: list[RetrievalResult] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> GeneratedResponse:
        """Execute the answer generation use case.

        Args:
            query: The user query.
            context: The retrieved context to use.
            sources: Optional list of source documents.
            temperature: LLM temperature setting.
            max_tokens: Maximum tokens for the response.

        Returns:
            GeneratedResponse with the answer and metadata.
        """
        start_time = perf_counter()

        # Format the prompts
        system_prompt = self.system_prompt_template.format(context=context)
        user_prompt = self.user_prompt_template.format(question=query.text)

        # Generate the answer
        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed_ms = (perf_counter() - start_time) * 1000

        return GeneratedResponse(
            query=query,
            answer=response.content.strip(),
            sources=sources or [],
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            generation_time_ms=elapsed_ms,
            metadata={
                "finish_reason": response.finish_reason,
                "temperature": temperature,
            },
        )

    async def execute_with_streaming(
        self,
        query: Query,
        context: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        """Execute answer generation with streaming response.

        Args:
            query: The user query.
            context: The retrieved context to use.
            temperature: LLM temperature setting.
            max_tokens: Maximum tokens for the response.

        Yields:
            String chunks of the generated answer.
        """
        # Format the prompts
        system_prompt = self.system_prompt_template.format(context=context)
        user_prompt = self.user_prompt_template.format(question=query.text)

        # Stream the answer
        async for chunk in self.llm.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk

    async def execute_with_custom_prompt(
        self,
        query: Query,
        context: str,
        system_prompt: str,
        user_prompt: str,
        sources: list[RetrievalResult] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> GeneratedResponse:
        """Execute with custom prompt templates.

        Args:
            query: The user query.
            context: The retrieved context to use.
            system_prompt: Custom system prompt (can use {context} placeholder).
            user_prompt: Custom user prompt (can use {question} placeholder).
            sources: Optional list of source documents.
            temperature: LLM temperature setting.
            max_tokens: Maximum tokens for the response.

        Returns:
            GeneratedResponse with the answer and metadata.
        """
        start_time = perf_counter()

        # Format the prompts with available placeholders
        formatted_system = system_prompt.format(context=context)
        formatted_user = user_prompt.format(question=query.text)

        # Generate the answer
        response = await self.llm.generate(
            prompt=formatted_user,
            system_prompt=formatted_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed_ms = (perf_counter() - start_time) * 1000

        return GeneratedResponse(
            query=query,
            answer=response.content.strip(),
            sources=sources or [],
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            generation_time_ms=elapsed_ms,
            metadata={
                "finish_reason": response.finish_reason,
                "temperature": temperature,
                "custom_prompt": True,
            },
        )
