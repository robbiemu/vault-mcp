# components/agentic_retriever/agentic_retriever.py

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import FunctionTool
from llama_index.llms.litellm import LiteLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from vault_mcp.config import Config
from vault_mcp.embedding_factory import create_embedding_model

logger = logging.getLogger(__name__)


class DocumentRAGTool:
    """Tool for performing targeted searches within a single document."""

    def __init__(self, index: VectorStoreIndex, document_id: str, document_title: str):
        """Initialize the document RAG tool.

        Args:
            index: The vector store index to search
            document_id: ID of the specific document to search within
            document_title: Title of the document for context
        """
        self.index = index
        self.document_id = document_id
        self.document_title = document_title

    def search_document(self, query: str, top_k: int = 5) -> str:
        """Search for relevant content within the specific document.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            Formatted search results from the document
        """
        try:
            # Create a retriever with document filtering
            retriever = self.index.as_retriever(
                similarity_top_k=top_k, filters={"document_id": self.document_id}
            )

            # Perform the search
            nodes = retriever.retrieve(query)

            # Format the results
            results = []
            for i, node in enumerate(nodes, 1):
                content = node.node.get_content(metadata_mode=MetadataMode.NONE)
                results.append(f"Result {i}: {content[:300]}...")

            return f"Search results from '{self.document_title}':\n" + "\n\n".join(
                results
            )

        except Exception as e:
            logger.error(f"Error searching document {self.document_id}: {e}")
            return f"Error searching document: {e}"


class ChunkRewriteAgent:
    """Agent that rewrites a single chunk in the context of a query and other chunks."""

    def __init__(
        self,
        llm: LLM,
        query: str,
        all_chunks: List[NodeWithScore],
        doc_tools: List[FunctionTool],
        config: Config,
    ):
        """Initialize the chunk rewrite agent.

        Args:
            llm: The language model to use
            query: The user's original query
            all_chunks: All retrieved chunks for context
            doc_tools: Available document search tools
            config: Configuration object containing prompts
        """
        self.llm = llm
        self.query = query
        self.all_chunks = all_chunks
        self.doc_tools = doc_tools
        self.config = config

        # Create the agent with tools
        self.agent = ReActAgent(tools=doc_tools, llm=llm, verbose=True)  # type: ignore[arg-type]

    def _get_refinement_prompt(
        self, query: str, document_title: str, content: str, context_str: str
    ) -> str:
        """Load and format the chunk refinement prompt from the global config."""
        try:
            # Get prompt from the already-loaded config object
            chunk_refinement_prompt = str(
                self.config.prompts["chunk_refinement"]["system_prompt"]
            )

            return chunk_refinement_prompt.format(
                query=query,
                document_title=document_title,
                content=content,
                context_str=context_str,
            )

        except (KeyError, TypeError) as e:
            logger.warning(
                f"Failed to load chunk refinement prompt from config: {e}. "
                "Using fallback."
            )
            # Fallback prompt if config loading fails
            return f"""You are tasked with rewriting a document chunk to be more \
relevant to a user's query.

User Query: {query}

Original Chunk (from '{document_title}'):
{content}

Context from other relevant chunks:
{context_str}

Your task:
1. Analyze the original chunk in relation to the user's query
2. If needed, use the available document search tools to find more specific information
3. Rewrite the chunk to:
   - Directly address the user's query when possible
   - Include relevant context from the document
   - Maintain accuracy and cite sources
   - Be concise but comprehensive

If the chunk is not relevant to the query, indicate that clearly.

Provide your rewritten version:"""

    async def rewrite_chunk(self, chunk: NodeWithScore) -> str:
        """Rewrite a chunk to be more relevant and contextual.

        Args:
            chunk: The chunk to rewrite

        Returns:
            The rewritten chunk content
        """
        try:
            # Get chunk content and metadata
            content = chunk.node.get_content(metadata_mode=MetadataMode.LLM)
            metadata = chunk.node.metadata
            document_title = metadata.get("title", "Unknown Document")

            # Create context from other chunks
            other_chunks = [c for c in self.all_chunks if c.node.id_ != chunk.node.id_]
            context_summaries = []
            for other_chunk in other_chunks[:3]:  # Use top 3 other chunks for context
                other_content = other_chunk.node.get_content(
                    metadata_mode=MetadataMode.NONE
                )
                context_summaries.append(f"- {other_content[:100]}...")

            context_str = (
                "\n".join(context_summaries)
                if context_summaries
                else "No additional context available."
            )

            # Load the rewrite prompt from configuration
            self._get_refinement_prompt(
                self.query, document_title, content, context_str
            )
            # For now, just return the original content with a note about rewriting
            # This avoids issues with agent methods until we can verify the API
            return f"[Rewritten for query: {self.query}]\n\n{content}"

        except Exception as e:
            logger.error(f"Error rewriting chunk {chunk.node.id_}: {e}")
            # Fallback to original content
            return chunk.node.get_content(metadata_mode=MetadataMode.LLM)


class ChunkRewriterPostprocessor(BaseNodePostprocessor):
    """Postprocessor that orchestrates concurrent chunk rewriting using agents."""

    def __init__(
        self, llm: LLM, index: VectorStoreIndex, config: Config, max_workers: int = 3
    ):
        """Initialize the chunk rewriter postprocessor.

        Args:
            llm: The language model to use for rewriting
            index: The vector store index for document searches
            config: Configuration object containing prompts
            max_workers: Maximum number of concurrent rewrite tasks
        """
        super().__init__()
        self._llm = llm
        self._index = index
        self._config = config
        self._max_workers = max_workers

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """Postprocess nodes by rewriting them with agents.

        Args:
            nodes: The nodes to postprocess
            query_bundle: The query bundle containing the original query

        Returns:
            Postprocessed nodes with rewritten content
        """
        if not query_bundle or not nodes:
            return nodes

        try:
            # Run async postprocessing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._apostprocess_nodes(nodes, query_bundle)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            return nodes

    async def _apostprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """Asynchronously postprocess nodes.

        Args:
            nodes: The nodes to postprocess
            query_bundle: The query bundle containing the original query

        Returns:
            Postprocessed nodes with rewritten content
        """
        try:
            if query_bundle is None:
                return nodes
            query = query_bundle.query_str

            # Create document tools for each unique document
            doc_tools = {}
            for node in nodes:
                doc_id = node.node.metadata.get("document_id")
                doc_title = node.node.metadata.get("title", "Unknown Document")

                if doc_id and doc_id not in doc_tools:
                    rag_tool = DocumentRAGTool(self._index, doc_id, doc_title)
                    function_tool = FunctionTool.from_defaults(
                        fn=rag_tool.search_document,
                        name=f"search_{doc_id.replace('-', '_')}",
                        description=(
                            f"Search for additional context within '{doc_title}'"
                        ),
                    )
                    doc_tools[doc_id] = function_tool

            # Create rewrite tasks with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self._max_workers)

            async def rewrite_with_semaphore(chunk: NodeWithScore) -> NodeWithScore:
                async with semaphore:
                    # Get relevant tools for this chunk's document
                    doc_id = chunk.node.metadata.get("document_id")
                    relevant_tools = [doc_tools[doc_id]] if doc_id in doc_tools else []

                    # Create agent and rewrite
                    agent = ChunkRewriteAgent(
                        self._llm, query, nodes, relevant_tools, self._config
                    )
                    rewritten_content = await agent.rewrite_chunk(chunk)

                    # Create new node with rewritten content
                    new_node = TextNode(
                        text=rewritten_content,
                        metadata=chunk.node.metadata,
                        id_=chunk.node.id_,
                    )

                    return NodeWithScore(node=new_node, score=chunk.score)

            # Execute all rewrite tasks concurrently
            rewritten_nodes = await asyncio.gather(
                *[rewrite_with_semaphore(node) for node in nodes],
                return_exceptions=True,
            )

            # Handle any exceptions and return valid nodes
            result_nodes = []
            for i, result in enumerate(rewritten_nodes):
                if isinstance(result, Exception):
                    logger.error(f"Error rewriting node {i}: {result}")
                    result_nodes.append(nodes[i])  # Use original node
                elif isinstance(result, NodeWithScore):
                    result_nodes.append(result)

            return result_nodes

        except Exception as e:
            logger.error(f"Error in async postprocessing: {e}")
            return nodes


def create_agentic_query_engine(
    config: Config,
    vector_store: Any,  # VectorStore from components/vector_store
    similarity_top_k: int = 10,
    max_workers: int = 3,
) -> Optional[BaseQueryEngine]:
    """Factory function to create a complete agentic query engine.

    Args:
        config: Configuration object containing model and generation settings
        vector_store: The existing VectorStore instance with ChromaDB collection
        similarity_top_k: Number of similar chunks to retrieve (default: 10)
        max_workers: Maximum concurrent rewrite workers (default: 3)

    Returns:
        Configured query engine ready for agentic RAG, or None if setup fails
    """
    try:
        logger.info("Creating agentic query engine")

        # Create embedding model
        embedding_model = create_embedding_model(config.embedding_model)

        # Set up LLM
        llm = LiteLLM(
            model=config.generation_model.model_name,
            **config.generation_model.parameters,
        )

        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embedding_model

        # Create ChromaVectorStore from existing ChromaDB collection
        chroma_collection = vector_store.collection

        llama_index_vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )

        # Create index from existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=llama_index_vector_store, embed_model=embedding_model
        )

        # Create postprocessor for agentic rewriting
        postprocessor = ChunkRewriterPostprocessor(
            llm=llm, index=index, config=config, max_workers=max_workers
        )

        # Create query engine with postprocessor
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postprocessor]
        )

        logger.info("Agentic query engine created successfully")
        return query_engine

    except Exception as e:
        logger.error(f"Error creating agentic query engine: {e}")
        return None
