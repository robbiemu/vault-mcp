# components/agentic_retriever/agentic_retriever.py

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from components.document_processing import (
    FullDocumentRetrievalTool,
    SectionHeadersTool,
    SectionRetrievalTool,
)
from components.embedding_system import create_embedding_model
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.litellm import LiteLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from vault_mcp.config import Config

logger = logging.getLogger(__name__)


class StaticContextPostprocessor(BaseNodePostprocessor):
    """Postprocessor that expands retrieved chunks to their full section context."""

    def __init__(self) -> None:
        """Initialize the static context postprocessor."""
        super().__init__()
        from components.document_processing.document_tools import DocumentReader

        self._document_reader = DocumentReader()

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """Postprocess nodes by expanding each to its full section context.

        Args:
            nodes: The nodes to postprocess
            query_bundle: The query bundle (unused in static mode)

        Returns:
            Postprocessed nodes with expanded section content
        """
        if not nodes:
            return nodes

        expanded_nodes = []
        seen_sections = {}  # For deduplication: key -> NodeWithScore

        for node in nodes:
            try:
                # Get the file path and character indices
                file_path = node.node.metadata.get("file_path")
                # Try to get character indices from metadata first (persisted),
                # then node attributes (fallback)
                start_char_idx = node.node.metadata.get("start_char_idx") or getattr(
                    node.node, "start_char_idx", None
                )
                end_char_idx = node.node.metadata.get("end_char_idx") or getattr(
                    node.node, "end_char_idx", None
                )

                logger.debug(
                    f"Node {node.node.id_}: file_path={file_path}, "
                    f"start_char_idx={start_char_idx}, end_char_idx={end_char_idx} "
                    f"(from metadata: {node.node.metadata.get('start_char_idx')}, "
                    f"{node.node.metadata.get('end_char_idx')})"
                )

                if not file_path or start_char_idx is None or end_char_idx is None:
                    logger.warning(
                        f"Node {node.node.id_} missing required metadata, "
                        f"keeping original (file_path={file_path}, "
                        f"start={start_char_idx}, end={end_char_idx})"
                    )
                    expanded_nodes.append(node)
                    continue

                # Get the full section context
                section_content, section_start_idx, section_end_idx = (
                    self._document_reader.get_enclosing_sections(
                        file_path, start_char_idx, end_char_idx
                    )
                )

                # Log expansion results
                original_length = len(node.node.get_content())
                expanded_length = len(section_content)
                logger.debug(
                    f"Node {node.node.id_} expansion: original={original_length} "
                    f"chars, expanded={expanded_length} chars "
                    f"({expanded_length / original_length:.1f}x), "
                    f"section range: {section_start_idx}-{section_end_idx}"
                )

                # Check if we actually expanded
                if section_content == node.node.get_content():
                    logger.warning(
                        f"Node {node.node.id_}: Section content identical to original "
                        f"chunk! This suggests section boundaries exactly match "
                        f"chunk boundaries."
                    )

                # Create unique key for deduplication.
                # Prefer deterministic keys based on file path and exact character range
                # to avoid duplicates caused by minor whitespace differences.
                if section_start_idx is not None and section_end_idx is not None:
                    section_key = f"{file_path}:{section_start_idx}:{section_end_idx}"
                else:
                    # Fallback to content hash when indices are unavailable
                    normalized = " ".join(
                        section_content.split()
                    )  # collapse whitespace
                    section_key = f"{file_path}:{hash(normalized)}"

                if section_key not in seen_sections:
                    # Create new node with section content and preserve character
                    # indices
                    new_node = TextNode(
                        text=section_content,
                        metadata=node.node.metadata.copy(),
                        id_=f"section_{node.node.id_}",
                    )

                    # Preserve the character indices from the expanded section
                    # Store in both metadata (for persistence) and node attributes
                    #  (for compatibility)
                    new_node.metadata["start_char_idx"] = section_start_idx
                    new_node.metadata["end_char_idx"] = section_end_idx
                    new_node.start_char_idx = section_start_idx
                    new_node.end_char_idx = section_end_idx

                    expanded_node = NodeWithScore(node=new_node, score=node.score)
                    seen_sections[section_key] = expanded_node
                    expanded_nodes.append(expanded_node)
                else:
                    # Section already seen, keep the higher scoring one
                    existing_node = seen_sections[section_key]
                    if (
                        node.score is not None
                        and existing_node.score is not None
                        and node.score > existing_node.score
                    ):
                        # Replace with higher scoring node
                        new_replacement_node = TextNode(
                            text=section_content,
                            metadata=node.node.metadata.copy(),
                            id_=f"section_{node.node.id_}",
                        )
                        # Preserve the character indices from the expanded section
                        # Store in both metadata (for persistence) and node
                        #  attributes (for compatibility)
                        new_replacement_node.metadata["start_char_idx"] = (
                            section_start_idx
                        )
                        new_replacement_node.metadata["end_char_idx"] = section_end_idx
                        new_replacement_node.start_char_idx = section_start_idx
                        new_replacement_node.end_char_idx = section_end_idx

                        seen_sections[section_key] = NodeWithScore(
                            node=new_replacement_node,
                            score=node.score,
                        )
                        # Update in expanded_nodes list
                        for i, exp_node in enumerate(expanded_nodes):
                            if exp_node is existing_node:
                                expanded_nodes[i] = seen_sections[section_key]
                                break

            except Exception as e:
                logger.error(f"Error expanding node {node.node.id_}: {e}")
                # Keep original node on error
                expanded_nodes.append(node)

        logger.info(
            f"Expanded {len(nodes)} chunks to {len(expanded_nodes)} "
            f"deduplicated sections"
        )
        return expanded_nodes


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

        # Debug: Log tool information
        logger.info(f"ChunkRewriteAgent initialized with {len(doc_tools)} tools:")
        for i, tool in enumerate(doc_tools):
            logger.info(
                f"  Tool {i + 1}: {tool.metadata.name} - {tool.metadata.description}"
            )

        if not doc_tools:
            logger.warning(
                "ChunkRewriteAgent initialized with NO TOOLS - this will "
                "cause tool calling failures!"
            )

        # Create the agent with tools and inherit global callback manager if available
        from llama_index.core import Settings

        # Safely get callback manager, defaulting to None if not set
        callback_manager = getattr(Settings, "callback_manager", None)

        # Only enable verbose mode when debugging is explicitly requested
        is_debugging = callback_manager is not None

        # Create agent with callback manager if it exists
        tools_list: List[Union[BaseTool, Callable[..., Any]]] = list(doc_tools)
        if is_debugging:
            self.agent = ReActAgent(
                tools=tools_list,
                llm=llm,
                verbose=True,
                callback_manager=callback_manager,
                max_iterations=self.config.retrieval.max_iterations,
            )
        else:
            # Fallback without callback manager
            self.agent = ReActAgent(
                tools=tools_list,
                llm=llm,
                verbose=False,
                max_iterations=self.config.retrieval.max_iterations,
            )

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
        """Rewrite a chunk to be more relevant and contextual."""
        try:
            # Get ONLY the clean text content, no metadata.
            content = chunk.node.get_content(metadata_mode=MetadataMode.NONE)
            metadata = chunk.node.metadata
            document_title = metadata.get("title", "Unknown Document")

            # Create context from other chunks
            other_chunks = [c for c in self.all_chunks if c.node.id_ != chunk.node.id_]
            context_summaries = []
            for other_chunk in other_chunks[:3]:
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
            prompt = self._get_refinement_prompt(
                self.query, document_title, content, context_str
            )

            # Use the agent's run method which returns an awaitable WorkflowHandler
            handler = self.agent.run(prompt)
            response = await handler
            return str(response)

        except Exception as e:
            logger.error(f"Error rewriting chunk {chunk.node.id_}: {e}")
            # Fallback to original content, but still clean.
            return chunk.node.get_content(metadata_mode=MetadataMode.NONE)


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
            # Run async postprocessing using modern asyncio.run()
            return asyncio.run(self._apostprocess_nodes(nodes, query_bundle))
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
            full_doc_tool = FullDocumentRetrievalTool()
            section_tool = SectionRetrievalTool()
            headers_tool = SectionHeadersTool()

            logger.info(f"Creating tools for {len(nodes)} nodes...")

            for i, node in enumerate(nodes):
                doc_id = node.node.metadata.get("document_id")
                doc_title = node.node.metadata.get("title", "Unknown Document")
                file_path = node.node.metadata.get("file_path")

                logger.debug(
                    f"Node {i + 1}/{len(nodes)} ({node.node.id_}): "
                    f"doc_id='{doc_id}', title='{doc_title}', file_path='{file_path}'"
                )
                logger.debug(
                    f"Node {i + 1} metadata keys: {list(node.node.metadata.keys())}"
                )

                if doc_id and doc_id not in doc_tools and file_path:
                    logger.info(
                        f"Creating tools for document: {doc_id} ({doc_title}) "
                        "at {file_path}"
                    )
                elif not doc_id:
                    logger.warning(
                        f"Node {i + 1} missing document_id - cannot create tools"
                    )
                elif not file_path:
                    logger.warning(
                        f"Node {i + 1} missing file_path - cannot create tools "
                        "(doc_id={doc_id})"
                    )
                elif doc_id in doc_tools:
                    logger.debug(f"Tools already exist for doc_id: {doc_id}")

                if doc_id and doc_id not in doc_tools and file_path:
                    # Create tool closures to capture the correct file_path
                    def make_full_doc_tool(path: str) -> Any:
                        def retrieve_full(dummy_arg: str = "unused") -> str:
                            return full_doc_tool.retrieve_full_document(path)

                        return retrieve_full

                    def make_section_tool(path: str) -> Any:
                        def get_sections(start_byte: int, end_byte: int) -> str:
                            return section_tool.get_enclosing_sections(
                                path, start_byte, end_byte
                            )

                        return get_sections

                    def make_headers_tool(path: str) -> Any:
                        def get_headers(dummy_arg: str = "unused") -> str:
                            return headers_tool.get_section_headers(path)

                        return get_headers

                    # Create tools for this document
                    full_doc_function_tool = FunctionTool.from_defaults(
                        fn=make_full_doc_tool(file_path),
                        name=f"read_full_{doc_id.replace('-', '_')}",
                        description=(
                            f"WARNING: Read the entire content of '{doc_title}'. "
                            f"This is a last resort for insufficient targeted methods."
                        ),
                    )

                    section_function_tool = FunctionTool.from_defaults(
                        fn=make_section_tool(file_path),
                        name=f"get_sections_{doc_id.replace('-', '_')}",
                        description=(
                            f"Get sections for context in '{doc_title}' using bytes. "
                            f"Primary tool for targeted context. Use start/end bytes."
                        ),
                    )

                    headers_function_tool = FunctionTool.from_defaults(
                        fn=make_headers_tool(file_path),
                        name=f"get_headers_{doc_id.replace('-', '_')}",
                        description=(
                            f"Get document structure overview for '{doc_title}'. "
                            f"Shows all section headers to help navigate the document."
                        ),
                    )

                    doc_tools[doc_id] = [
                        full_doc_function_tool,
                        section_function_tool,
                        headers_function_tool,
                    ]

            # Log summary of tool creation
            total_tools_created = sum(len(tools) for tools in doc_tools.values())
            logger.info(
                f"Tool creation summary: Created {total_tools_created} tools "
                "for {len(doc_tools)} documents"
            )
            if not doc_tools:
                logger.error(
                    "CRITICAL: No tools were created! This will cause all "
                    "agents to fail."
                )
            else:
                for doc_id, tools in doc_tools.items():
                    tool_names = [tool.metadata.name for tool in tools]
                    logger.info(f"  Document {doc_id}: {tool_names}")

            # Create rewrite tasks with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self._max_workers)

            async def rewrite_with_semaphore(chunk: NodeWithScore) -> NodeWithScore:
                async with semaphore:
                    try:
                        # Get relevant tools for this chunk's document
                        doc_id = chunk.node.metadata.get("document_id")
                        relevant_tools = doc_tools.get(doc_id, [])

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
                    except Exception as e:
                        logger.error(f"Error rewriting chunk {chunk.node.id_}: {e}")
                        # Create fallback node with original content and error metadata
                        fallback_node = TextNode(
                            text=chunk.node.get_content(
                                metadata_mode=MetadataMode.NONE
                            ),
                            metadata={**chunk.node.metadata, "rewrite_error": str(e)},
                            id_=chunk.node.id_,
                        )
                        return NodeWithScore(node=fallback_node, score=chunk.score)

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
    max_workers: int = 3,
) -> Optional[Union[BaseQueryEngine, ExpandedSourceQueryEngine]]:
    """Factory function to create a query engine with configurable post-processing.

    Args:
        config: Configuration object containing model and generation settings
        vector_store: The existing VectorStore instance with ChromaDB collection
        similarity_top_k: Number of similar chunks to retrieve (default: 10)
        max_workers: Maximum concurrent rewrite workers (default: 3)

    Returns:
        Configured query engine ready for RAG, or None if setup fails
    """
    try:
        logger.info(f"Creating query engine in '{config.retrieval.mode}' mode")

        # Create embedding model
        embedding_model = create_embedding_model(config.embedding_model)

        # Set global embedding model setting
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

        # Factory logic: Choose post-processor based on retrieval mode
        postprocessor: Union[ChunkRewriterPostprocessor, StaticContextPostprocessor]
        if config.retrieval.mode == "agentic":
            # Set up LLM for agentic mode
            if config.generation_model is None:
                raise ValueError("generation_model is required for agentic mode")

            llm = LiteLLM(
                model=config.generation_model.model_name,
                **config.generation_model.parameters,
            )
            Settings.llm = llm

            # Enable debugging if llamaindex_debugging is set
            if config.retrieval.llamaindex_debugging:
                from llama_index.core.callbacks import (
                    CallbackManager,
                    LlamaDebugHandler,
                )
                from llama_index.core.callbacks.base import BaseCallbackHandler
                from llama_index.core.callbacks.schema import CBEventType

                class ReActVerboseHandler(BaseCallbackHandler):
                    """Custom handler to show ReAct agent reasoning steps."""

                    def __init__(self) -> None:
                        super().__init__([], [])

                    def _should_log(self, event_type: CBEventType) -> bool:
                        return True

                    def start_trace(self, trace_id: Optional[str] = None) -> None:
                        """Start a trace - no-op implementation."""
                        pass

                    def end_trace(
                        self,
                        trace_id: Optional[str] = None,
                        trace_map: Optional[dict] = None,
                    ) -> None:
                        """End a trace - no-op implementation."""
                        pass

                    def on_event_start(
                        self,
                        event_type: CBEventType,
                        payload: Optional[Dict[str, Any]] = None,
                        event_id: str = "",
                        parent_id: str = "",
                        **kwargs: Any,
                    ) -> str:
                        if event_type == CBEventType.LLM:
                            # Debug: print payload structure to understand format
                            print(f"\n🔍 LLM Start - Payload: {payload}")

                        if event_type == CBEventType.FUNCTION_CALL:
                            print(f"\n🔍 Function Call Start - Payload: {payload}")

                        if event_type == CBEventType.AGENT_STEP:
                            print(f"\n🔍 Agent Step Start - Payload: {payload}")
                        return event_id or ""

                    def on_event_end(
                        self,
                        event_type: CBEventType,
                        payload: Optional[Dict[str, Any]] = None,
                        event_id: str = "",
                        **kwargs: Any,
                    ) -> None:
                        if event_type == CBEventType.LLM:
                            print(f"\n🔍 LLM End - Payload: {payload}")
                            # Try different possible payload structures
                            response = ""
                            if payload:
                                response = (
                                    payload.get("response", "")
                                    or payload.get("output", "")
                                    or str(payload.get("result", ""))
                                )
                            if response:
                                print(f"\n💬 LLM Output:\n{response}")

                        if event_type == CBEventType.FUNCTION_CALL:
                            print(f"\n🔍 Function Call End - Payload: {payload}")

                        if event_type == CBEventType.AGENT_STEP:
                            print(f"\n🔍 Agent Step End - Payload: {payload}")

                # Create handlers for comprehensive debugging
                verbose_handler = ReActVerboseHandler()
                llama_debug = LlamaDebugHandler(print_trace_on_end=True)
                callback_manager = CallbackManager([verbose_handler, llama_debug])
                Settings.callback_manager = callback_manager

                # Note: Settings.debug is not available in current LlamaIndex version
                # General debug mode is enabled via callback managers instead

                # Set LlamaIndex-specific loggers to DEBUG level
                logging.getLogger("llama_index").setLevel(logging.DEBUG)
                logging.getLogger("llama_index.core").setLevel(logging.DEBUG)
                logging.getLogger("llama_index.core.agent").setLevel(logging.DEBUG)
                logging.getLogger("llama_index.core.query_engine").setLevel(
                    logging.DEBUG
                )

                logger.info(
                    "LlamaIndex debugging is enabled - you'll see detailed "
                    "turn-by-turn ReAct agent output"
                )

            # Create agentic postprocessor
            postprocessor = ChunkRewriterPostprocessor(
                llm=llm, index=index, config=config, max_workers=max_workers
            )
            logger.info("Using agentic postprocessor with LLM rewriting")

        elif config.retrieval.mode == "static":
            # Create static postprocessor (no LLM required)
            postprocessor = StaticContextPostprocessor()

            # Set a dummy LLM to prevent LlamaIndex from trying to initialize OpenAI
            # This is needed because index.as_query_engine() requires an LLM in Settings
            from llama_index.core.llms import MockLLM

            Settings.llm = MockLLM()

            logger.info("Using static postprocessor with section expansion")

        else:
            raise ValueError(
                f"Invalid retrieval mode: {config.retrieval.mode}. "
                f"Must be 'agentic' or 'static'"
            )

        # Create base query engine with the selected postprocessor
        base_query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postprocessor]
        )

        # For static mode, wrap to ensure expanded nodes are returned in source_nodes
        query_engine: Union[BaseQueryEngine, ExpandedSourceQueryEngine]
        if config.retrieval.mode == "static":
            query_engine = ExpandedSourceQueryEngine(base_query_engine, postprocessor)
            logger.info(
                "Wrapped with ExpandedSourceQueryEngine to return expanded sections"
            )
        else:
            query_engine = base_query_engine

        logger.info(
            f"Query engine created successfully in '{config.retrieval.mode}' mode"
        )
        return query_engine

    except Exception as e:
        logger.error(f"Error creating query engine: {e}")
        return None
