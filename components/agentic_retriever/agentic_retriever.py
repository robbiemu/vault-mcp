from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional, Union

from components.document_processing import (
    DocumentReader,
)
from components.embedding_system import create_embedding_model
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.litellm import LiteLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from shared.config import Config

from .logging_handler import ReActVerboseHandler

logger = logging.getLogger(__name__)


class StaticContextPostprocessor(BaseNodePostprocessor):
    """Postprocessor that expands retrieved chunks to their full section context."""

    def __init__(self) -> None:
        """Initialize the static context postprocessor."""
        super().__init__()
        self._document_reader = DocumentReader()

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """Postprocess nodes by expanding each to its full section context."""
        if not nodes:
            return nodes

        expanded_nodes = []
        seen_sections = {}  # For deduplication: key -> NodeWithScore

        for node in nodes:
            try:
                metadata = node.node.metadata
                file_path = metadata.get("file_path")
                start_char_idx = metadata.get("start_char_idx")
                end_char_idx = metadata.get("end_char_idx")

                if not file_path or start_char_idx is None or end_char_idx is None:
                    logger.warning(
                        f"Node {node.node.id_} is missing required metadata for "
                        "expansion. Keeping original node."
                    )
                    expanded_nodes.append(node)
                    continue

                section_content, section_start_idx, section_end_idx = (
                    self._document_reader.get_enclosing_sections(
                        file_path, start_char_idx, end_char_idx
                    )
                )
                section_key = f"{file_path}:{section_start_idx}:{section_end_idx}"

                if section_key not in seen_sections:
                    new_node = TextNode(
                        text=section_content,
                        metadata=node.node.metadata.copy(),
                        id_=f"section_{node.node.id_}",
                    )
                    new_node.metadata["start_char_idx"] = section_start_idx
                    new_node.metadata["end_char_idx"] = section_end_idx
                    new_node.start_char_idx = section_start_idx
                    new_node.end_char_idx = section_end_idx
                    expanded_node = NodeWithScore(node=new_node, score=node.score)
                    seen_sections[section_key] = expanded_node
                    expanded_nodes.append(expanded_node)
                else:
                    existing_node = seen_sections[section_key]
                    if (
                        node.score is not None
                        and existing_node.score is not None
                        and node.score > existing_node.score
                    ):
                        for i, exp_node in enumerate(expanded_nodes):
                            if exp_node is existing_node:
                                expanded_nodes[i] = NodeWithScore(
                                    node=exp_node.node, score=node.score
                                )
                                seen_sections[section_key] = expanded_nodes[i]
                                break
            except Exception as e:
                logger.error(
                    f"Error expanding node {node.node.id_}: {e}", exc_info=True
                )
                expanded_nodes.append(node)

        logger.info(
            f"Expanded {len(nodes)} chunks to {len(expanded_nodes)} "
            f"deduplicated sections"
        )
        return expanded_nodes


class ChunkRewriteAgent:
    """Agent that rewrites a single chunk using a shared, powerful toolset."""

    def __init__(
        self,
        llm: LLM,
        query: str,
        available_files_str: str,
        shared_tools: List[FunctionTool],
        config: Config,
        callback_manager: Optional[Any] = None,
    ):
        """Initialize the chunk rewrite agent with a shared toolset."""
        self.llm = llm
        self.query = query
        self.available_files_str = available_files_str
        self.config = config
        tools_list: List[Union[BaseTool, Callable[..., Any]]] = list(shared_tools)
        self.agent = ReActAgent(
            tools=tools_list,
            llm=llm,
            verbose=callback_manager is not None,
            callback_manager=callback_manager,
            max_iterations=self.config.retrieval.max_iterations,
        )

    def _get_refinement_prompt(
        self, query: str, document_title: str, content: str, context_str: str
    ) -> str:
        """Load and format the chunk refinement prompt."""
        try:
            chunk_refinement_prompt = str(
                self.config.prompts["chunk_refinement"]["system_prompt"]
            )
            return chunk_refinement_prompt.format(
                query=query,
                document_title=document_title,
                content=content,
                context_str=context_str,
                available_files=self.available_files_str,
            )
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to load prompt from config: {e}. Using fallback.")
            return f"""You are an expert synthesizer of technical documentation...
**User Query:** {query}
**Available Files (use the number as the `file_index`):**
{self.available_files_str}
**Seed Chunk (from '{document_title}'):**
{content}
**Context from other retrieved chunks:**
{context_str}
Your Task: ..."""

    def _get_salvage_history(self) -> str:
        """
        Retrieves and formats the agent's conversation history for salvaging.
        Returns all messages EXCEPT the final one, which often contains the error.
        """
        try:
            # The memory is attached to the agent instance
            memory = getattr(self.agent, "memory", None)
            if not memory:
                return "No history available (agent memory not found)."

            # get_all() returns a list of ChatMessage objects
            all_messages = memory.get_all()
            if not all_messages:
                return "History is empty."

            # Exclude the last message, which is often the failed/empty one
            history_messages = all_messages[:-1]
            if not history_messages:
                return "No valid history to salvage (only one message)."

            # Format the structured messages into a clean string for the prompt
            formatted_history = []
            for msg in history_messages:
                role_str = msg.role.value.upper()
                # Ensure content is a string, handling potential None or other types
                content_str = str(msg.content) if msg.content else ""
                formatted_history.append(f"--- Role: {role_str} ---\n{content_str}")

            return "\n\n".join(formatted_history)

        except Exception as e:
            logger.error(f"Error processing agent memory for salvage: {e}")
            return f"Failed to retrieve history due to an error: {e}"

    async def rewrite_chunk(self, chunk: NodeWithScore, context_str: str) -> str:
        """Asynchronously rewrite a chunk using a two-tier fallback system."""
        content = chunk.node.get_content(metadata_mode=MetadataMode.NONE)

        try:
            metadata = chunk.node.metadata
            document_title = metadata.get(
                "title", metadata.get("file_path", "Unknown Document")
            )

            # The context_str is now passed in from the orchestrator
            prompt = self._get_refinement_prompt(
                self.query, document_title, content, context_str
            )

            response = await self.agent.run(prompt)
            return str(response)

        except Exception as e:
            # Tier 2: If agent fails for ANY reason, try the salvage logic
            logger.warning(
                f"Agent failed for chunk {chunk.node.id_} with error: {e}. "
                "Attempting to salvage with a context wrap prompt."
            )
            try:
                history = self._get_salvage_history()

                # Use a default prompt template for safety
                wrap_prompt_template = self.config.prompts.get(
                    "chunk_refinement", {}
                ).get(
                    "context_wrap",
                    "Based on the following conversation history, please provide the "
                    "final rewritten chunk for the user's query.\n\nQuery: "
                    f"{self.query}"
                    f"\n\nOriginal Chunk: {content}\n\nConversation History:\n{history}"
                    "\n\nFinal Answer:",
                )

                wrap_prompt = wrap_prompt_template.format(
                    query=self.query, content=content, history=history
                )

                # Make a direct, one-shot ASYNC call to the LLM
                final_response = await self.llm.achat(
                    messages=[ChatMessage(role="user", content=wrap_prompt)]
                )

                logger.info(
                    f"Successfully salvaged chunk {chunk.node.id_} using context wrap."
                )
                return str(final_response)

            except Exception as salvage_e:
                # Tier 3: If the salvage ALSO fails, fall back to original content
                logger.error(
                    f"Context wrap salvage failed for chunk {chunk.node.id_}:"
                    f" {salvage_e}. "
                    "Falling back to original content."
                )
                return content


class ChunkRewriterPostprocessor(BaseNodePostprocessor):
    """Orchestrates sequential chunk rewriting using a shared, indexed toolset."""

    def __init__(
        self,
        llm: LLM,
        config: Config,
        callback_manager: Optional[Any] = None,
    ):
        """Initialize the chunk rewriter postprocessor."""
        super().__init__()
        self._llm = llm
        self._config = config
        self._callback_manager = callback_manager
        self._document_reader = DocumentReader()

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """Synchronous entry point that runs the entire async rewriting process."""
        if not query_bundle or not nodes:
            return nodes

        # Use asyncio.run() once from the sync context to run the async orchestrator.
        return asyncio.run(self._apostprocess_nodes(nodes, query_bundle))

    # In: components/agentic_retriever/agentic_retriever.py

    async def _apostprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """
        Orchestrates the rewrite of EACH retrieved chunk, providing the other chunks
        as context to the agent for each operation.
        """
        if not query_bundle:
            return nodes
        query = query_bundle.query_str

        # 1. Setup: Create the file list and tool functions. This is done once.
        file_paths: set[str] = set()
        for node in nodes:
            file_path = node.node.metadata.get("file_path")
            if isinstance(file_path, str):
                file_paths.add(file_path)
        available_files = sorted(list(file_paths))
        index_to_path_map = {i + 1: path for i, path in enumerate(available_files)}
        numbered_list_str = "\n".join(
            f"{i}: `{path}`" for i, path in index_to_path_map.items()
        )
        logger.info(f"Agent will have access to these files:\n{numbered_list_str}")

        def get_path_from_index(file_index: int) -> str:
            path = index_to_path_map.get(file_index)
            if not path:
                valid_indices = ", ".join(map(str, index_to_path_map.keys()))
                raise ValueError(
                    f"Invalid file_index: {file_index}. Valid indices are: "
                    f"{valid_indices}."
                )
            return path

        def read_full_document_by_index(file_index: int) -> str:
            return self._document_reader.read_full_document(
                get_path_from_index(file_index)
            )

        def get_sections_by_index(
            file_index: int, start_char_idx: int, end_char_idx: int
        ) -> str:
            content, _, _ = self._document_reader.get_enclosing_sections(
                get_path_from_index(file_index), start_char_idx, end_char_idx
            )
            return content

        def get_headers_by_index(file_index: int) -> str:
            headers_str, _ = self._document_reader.get_section_headers(
                get_path_from_index(file_index)
            )
            return headers_str

        def final_answer_tool_func(answer: str) -> str:
            """Returns the final, synthesized answer for the chunk."""
            return answer

        final_answer_tool = FunctionTool.from_defaults(
            fn=final_answer_tool_func,
            name="final_answer",
            description=(
                "Use this tool ONLY ONCE when you have the complete, final, "
                "rewritten chunk. The input is the full text of your final answer."
            ),
            return_direct=True,
        )

        shared_tools = [
            FunctionTool.from_defaults(
                fn=read_full_document_by_index,
                name="read_full_document",
                description="...",
            ),
            FunctionTool.from_defaults(
                fn=get_sections_by_index, name="get_sections", description="..."
            ),
            FunctionTool.from_defaults(
                fn=get_headers_by_index, name="get_headers", description="..."
            ),
            final_answer_tool,
        ]

        # 2. Sequentially rewrite EACH chunk using a loop
        rewritten_nodes = []
        logger.info(f"Starting sequential rewrite of {len(nodes)} chunks...")
        for i, chunk in enumerate(nodes):
            logger.info(
                f"Rewriting chunk {i + 1}/{len(nodes)} (ID: {chunk.node.id_})..."
            )

            # For each chunk, prepare the context from all other chunks.
            other_chunks = [c for c in nodes if c.node.id_ != chunk.node.id_]
            context_summaries = []
            # Limit to the top 3-4 other chunks to avoid excessive prompt length
            for other in other_chunks[:4]:
                other_content = other.node.get_content(metadata_mode=MetadataMode.NONE)
                context_summaries.append(f"- {other_content[:200]}...")
            context_str = (
                "\n".join(context_summaries)
                if context_summaries
                else "No additional context from other chunks provided."
            )

            try:
                agent = ChunkRewriteAgent(
                    self._llm,
                    query,
                    numbered_list_str,
                    shared_tools,
                    self._config,
                    callback_manager=self._callback_manager,
                )
                # The agent now receives the context_str in its prompt via rewrite_chunk
                rewritten_content = await agent.rewrite_chunk(chunk, context_str)
                new_node = TextNode(
                    text=rewritten_content,
                    metadata=chunk.node.metadata,
                    id_=chunk.node.id_,
                )
                rewritten_nodes.append(NodeWithScore(node=new_node, score=chunk.score))
            except Exception as e:
                logger.error(
                    f"Error rewriting chunk {chunk.node.id_}: {e}", exc_info=True
                )
                rewritten_nodes.append(chunk)

        logger.info("Sequential rewrite complete.")
        return rewritten_nodes


class ExpandedSourceQueryEngine(BaseQueryEngine):
    """Wrapper that ensures postprocessed nodes are returned in source_nodes."""

    def __init__(
        self, base_query_engine: BaseQueryEngine, postprocessor: BaseNodePostprocessor
    ):
        super().__init__(callback_manager=None)
        self.base_query_engine = base_query_engine
        self.postprocessor = postprocessor

    def _query(self, query_bundle: QueryBundle) -> Any:
        """Query implementation required by BaseQueryEngine."""
        return self.query(query_bundle.query_str)

    async def _aquery(self, query_bundle: QueryBundle) -> Any:
        """Async query implementation required by BaseQueryEngine."""
        return self.query(query_bundle.query_str)

    def _get_prompt_modules(self) -> Any:
        """Get prompt modules - required by BaseQueryEngine."""
        return {}

    def query(self, query_str: str) -> Any:
        # Get the normal response
        response = self.base_query_engine.query(query_str)

        # If we have source nodes, reprocess them to get expanded versions
        if hasattr(response, "source_nodes") and response.source_nodes:
            from llama_index.core.schema import QueryBundle

            query_bundle = QueryBundle(query_str=query_str)

            # Apply postprocessing to get expanded nodes
            expanded_nodes = self.postprocessor.postprocess_nodes(
                response.source_nodes, query_bundle
            )

            # Replace the source nodes with expanded ones
            response.source_nodes = expanded_nodes

        return response


def create_agentic_query_engine(
    config: Config,
    vector_store: Any,  # VectorStore from components/vector_store
    similarity_top_k: int = 10,
) -> Optional[Union[BaseQueryEngine, ExpandedSourceQueryEngine]]:
    """Factory function to create a query engine with configurable post-processing."""
    try:
        logger.info(f"Creating query engine in '{config.retrieval.mode}' mode")

        embedding_model = create_embedding_model(config.embedding_model)
        Settings.embed_model = embedding_model

        chroma_collection = vector_store.collection
        llama_index_vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=llama_index_vector_store, embed_model=embedding_model
        )

        # The core logic for creating the postprocessor remains the same
        postprocessor: Union[ChunkRewriterPostprocessor, StaticContextPostprocessor]
        if config.retrieval.mode == "agentic":
            if config.generation_model is None:
                raise ValueError("generation_model is required for agentic mode")

            llm_parameters = config.generation_model.parameters or {}
            llm = LiteLLM(model=config.generation_model.model_name, **llm_parameters)
            Settings.llm = llm

            callback_manager = None
            if config.retrieval.llamaindex_debugging:
                verbose_handler = ReActVerboseHandler()
                llama_debug = LlamaDebugHandler(print_trace_on_end=True)
                callback_manager = CallbackManager([verbose_handler, llama_debug])
                Settings.callback_manager = callback_manager
                logging.getLogger("llama_index").setLevel(logging.DEBUG)
                logger.info("LlamaIndex debugging is enabled.")

            postprocessor = ChunkRewriterPostprocessor(
                llm=llm,
                config=config,
                callback_manager=callback_manager,
            )
            logger.info("Using agentic postprocessor with shared, file-aware tools.")

        elif config.retrieval.mode == "static":
            postprocessor = StaticContextPostprocessor()
            from llama_index.core.llms import MockLLM

            Settings.llm = MockLLM()
            logger.info("Using static postprocessor with section expansion.")
        else:
            raise ValueError(f"Invalid retrieval mode: {config.retrieval.mode}")

        retriever = index.as_retriever(similarity_top_k=similarity_top_k)

        query_engine: BaseQueryEngine
        if config.retrieval.mode == "agentic":
            # For agentic mode, we build the engine manually to set the response_mode
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[postprocessor],
                response_mode=ResponseMode.NO_TEXT,  # Your key insight
            )
            logger.info(
                "Created agentic query engine with ResponseMode.NO_TEXT to disable "
                "final synthesis."
            )
        else:
            # For static mode, the default engine is fine, but we wrap it
            base_query_engine = index.as_query_engine(
                similarity_top_k=similarity_top_k, node_postprocessors=[postprocessor]
            )
            query_engine = ExpandedSourceQueryEngine(base_query_engine, postprocessor)

        logger.info(
            f"Query engine created successfully in '{config.retrieval.mode}' mode"
        )
        return query_engine

    except Exception as e:
        logger.error(f"Error creating query engine: {e}", exc_info=True)
        return None
