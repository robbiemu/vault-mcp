import json
import logging
from typing import Any, List

from components.embedding_system import (
    OpenAIEndpointEmbedding,
)
from vault_mcp.config import EmbeddingModelConfig

logger = logging.getLogger(__name__)


class E5InstructWrapper(OpenAIEndpointEmbedding):
    """
    Concrete implementation of a custom wrapper for E5 instruction-tuned models
    accessed via an OpenAI-compatible endpoint.
    """

    def __init__(self, config: EmbeddingModelConfig, **kwargs: Any):
        super().__init__(
            model_name=config.model_name,
            endpoint_url=config.endpoint_url or "",
            api_key=config.api_key or "",
            **kwargs,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Overrides the base method to parse the query string and apply the
        correct instruction format for E5-instruct models.
        """
        default_instruction = (
            "Given a user query, retrieve the most relevant document chunks."
        )
        final_query = query
        instruction = default_instruction

        try:
            payload = json.loads(query)
            if (
                isinstance(payload, dict)
                and "instruction" in payload
                and "query" in payload
            ):
                instruction = payload["instruction"]
                final_query = payload["query"]
                logger.debug(f"Using custom instruction for embedding: '{instruction}'")
        except (json.JSONDecodeError, TypeError):
            logger.debug("No custom instruction found, using default.")

        formatted_query = f"Instruct: {instruction}\nQuery: {final_query}"
        return super()._get_query_embedding(formatted_query)
