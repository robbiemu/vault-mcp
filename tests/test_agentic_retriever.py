from unittest.mock import AsyncMock, Mock, patch
from pydantic import ConfigDict

import pytest
from components.agentic_retriever.agentic_retriever import (
    ChunkRewriteAgent,
    ChunkRewriterPostprocessor,
    ExpandedSourceQueryEngine,
    StaticContextPostprocessor,
    create_agentic_query_engine,
)
from components.vector_store.vector_store import VectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from shared.config import (
    Config,
    EmbeddingModelConfig,
    GenerationModelConfig,
    RetrievalConfig,
)


# FIX: Create a proper subclass of MockLLM that implements the streaming interface.
class StreamingMockLLM(MockLLM):
    def stream_chat(self, *args, **kwargs):
        yield ChatResponse(
            message=ChatMessage(role="assistant", content="Rewritten content")
        )

    model_config = ConfigDict(arbitrary_types_allowed=True)


@pytest.fixture
def mock_llm():
    # FIX: Instantiate the StreamingMockLLM directly.
    llm = StreamingMockLLM()
    llm.callback_manager = Mock()
    return llm


@pytest.fixture
def mock_config():
    config = Mock(spec=Config)
    config.retrieval = Mock(spec=RetrievalConfig)
    config.retrieval.llamaindex_debugging = False
    config.retrieval.max_iterations = 10
    config.prompts = {
        "chunk_refinement": {
            "system_prompt": "Rewrite: {query} {document_title} {content} {context_str}"
        }
    }
    # FIX: Ensure generation_model and its parameters are correctly mocked.
    generation_model_mock = Mock(spec=GenerationModelConfig)
    generation_model_mock.model_name = "test-model"
    generation_model_mock.parameters = {}  # Must be a dict
    config.generation_model = generation_model_mock

    config.server = Mock()
    config.server.default_query_limit = 5
    config.embedding_model = Mock()
    return config


@pytest.fixture
def mock_nodes():
    node1 = NodeWithScore(
        node=TextNode(
            text="Node 1 content",
            id_="node1",
            metadata={"file_path": "file1.md", "title": "Doc 1", "document_id": "doc1"},
        ),
        score=0.8,
    )
    node2 = NodeWithScore(
        node=TextNode(
            text="Node 2 content",
            id_="node2",
            metadata={"file_path": "file2.md", "title": "Doc 2", "document_id": "doc2"},
        ),
        score=0.7,
    )
    return [node1, node2]


@pytest.fixture
def mock_vector_store_instance(temp_vault_dir):
    with patch(
        "components.vector_store.vector_store.create_embedding_model"
    ) as mock_create_embedding:
        mock_embedding = Mock()
        mock_embedding.encode.side_effect = lambda chunks: [[0.1] * 768 for _ in chunks]
        mock_create_embedding.return_value = mock_embedding

        vector_store = VectorStore(
            embedding_config=EmbeddingModelConfig(
                provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
            ),
            persist_directory=str(temp_vault_dir / "test_chroma"),
            collection_name="test_vault_docs",
        )
        try:
            yield vector_store
        finally:
            vector_store.clear_all()


class TestChunkRewriteAgent:
    def test_init(self, mock_llm, mock_config, mock_nodes):
        # FIX: Patch the correct target for ReActAgent
        with patch("components.agentic_retriever.agentic_retriever.ReActAgent"):
            agent = ChunkRewriteAgent(mock_llm, "query", mock_nodes, [], mock_config)
            assert agent.llm == mock_llm
            assert agent.query == "query"
            assert agent.all_chunks == mock_nodes
            assert agent.doc_tools == []
            assert agent.config == mock_config
            assert isinstance(agent.agent, Mock)

    def test_get_refinement_prompt_success(self, mock_llm, mock_config):
        with patch("components.agentic_retriever.agentic_retriever.ReActAgent"):
            agent = ChunkRewriteAgent(mock_llm, "test query", [], [], mock_config)
            prompt = agent._get_refinement_prompt(
                "test query", "Test Doc", "content", "context"
            )
            assert "Rewrite: test query Test Doc content context" in prompt

    def test_get_refinement_prompt_fallback(self, mock_llm, mock_config):
        mock_config.prompts = {}
        with patch("components.agentic_retriever.agentic_retriever.ReActAgent"):
            agent = ChunkRewriteAgent(mock_llm, "test query", [], [], mock_config)
            prompt = agent._get_refinement_prompt(
                "test query", "Test Doc", "content", "context"
            )
            assert "You are tasked with rewriting a document chunk" in prompt

    @pytest.mark.asyncio
    async def test_rewrite_chunk_success(self, mock_llm, mock_config, mock_nodes):
        with patch(
            "components.agentic_retriever.agentic_retriever.ReActAgent"
        ) as MockReActAgent:
            mock_agent_instance = MockReActAgent.return_value

            # The agent.run() method returns an awaitable.
            import asyncio

            future = asyncio.Future()
            future.set_result("Rewritten content")
            mock_agent_instance.run.return_value = future

            agent = ChunkRewriteAgent(mock_llm, "query", mock_nodes, [], mock_config)
            result = await agent.rewrite_chunk(mock_nodes[0])

            assert result == "Rewritten content"
            # FIX: Assert the call on the mock instance directly.
            mock_agent_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_rewrite_chunk_error_fallback(
        self, mock_llm, mock_config, mock_nodes
    ):
        # FIX: Patch the correct target.
        with patch(
            "components.agentic_retriever.agentic_retriever.ReActAgent"
        ) as MockReActAgent:
            mock_agent_instance = MockReActAgent.return_value
            mock_agent_instance.run.side_effect = Exception("Agent error")

            agent = ChunkRewriteAgent(mock_llm, "query", mock_nodes, [], mock_config)
            result = await agent.rewrite_chunk(mock_nodes[0])

            # FIX: The exception is caught, and the original content should be returned.
            assert result == mock_nodes[0].node.get_content(
                metadata_mode=MetadataMode.NONE
            )


class TestChunkRewriterPostprocessor:
    def test_init(self, mock_llm, mock_config, mock_vector_store_instance):
        postprocessor = ChunkRewriterPostprocessor(
            mock_llm, Mock(spec=VectorStoreIndex), mock_config
        )
        assert postprocessor._llm == mock_llm
        assert postprocessor._config == mock_config

    @pytest.mark.asyncio
    async def test_apostprocess_nodes_no_query_bundle(
        self, mock_llm, mock_config, mock_nodes, mock_vector_store_instance
    ):
        postprocessor = ChunkRewriterPostprocessor(
            mock_llm, Mock(spec=VectorStoreIndex), mock_config
        )
        result = await postprocessor._apostprocess_nodes(mock_nodes, query_bundle=None)
        assert result == mock_nodes

    @pytest.mark.asyncio
    async def test_apostprocess_nodes_no_nodes(
        self, mock_llm, mock_config, mock_vector_store_instance
    ):
        postprocessor = ChunkRewriterPostprocessor(
            mock_llm, Mock(spec=VectorStoreIndex), mock_config
        )
        query_bundle = QueryBundle(query_str="test query")
        result = await postprocessor._apostprocess_nodes([], query_bundle)
        assert result == []

    @pytest.mark.asyncio
    async def test_apostprocess_nodes_success(
        self, mock_llm, mock_config, mock_nodes, mock_vector_store_instance
    ):
        postprocessor = ChunkRewriterPostprocessor(
            mock_llm, Mock(spec=VectorStoreIndex), mock_config
        )
        query_bundle = QueryBundle(query_str="test query")

        with patch(
            "components.agentic_retriever.agentic_retriever.ChunkRewriteAgent"
        ) as MockAgent:
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.rewrite_chunk = AsyncMock(
                return_value="Rewritten content"
            )

            result = await postprocessor._apostprocess_nodes(mock_nodes, query_bundle)

            assert len(result) == len(mock_nodes)
            assert all(isinstance(node, NodeWithScore) for node in result)
            assert result[0].node.get_content() == "Rewritten content"
            MockAgent.assert_called()
            mock_agent_instance.rewrite_chunk.assert_called()

    @pytest.mark.asyncio
    async def test_apostprocess_nodes_error_handling(
        self, mock_llm, mock_config, mock_nodes, mock_vector_store_instance
    ):
        postprocessor = ChunkRewriterPostprocessor(
            mock_llm, Mock(spec=VectorStoreIndex), mock_config
        )
        query_bundle = QueryBundle(query_str="test query")

        with patch(
            "components.agentic_retriever.agentic_retriever.ChunkRewriteAgent"
        ) as MockAgent:
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.rewrite_chunk = AsyncMock(
                side_effect=Exception("Rewrite error")
            )

            result = await postprocessor._apostprocess_nodes(mock_nodes, query_bundle)

            assert len(result) == len(mock_nodes)
            assert all(
                node.node.metadata.get("rewrite_error") == "Rewrite error"
                for node in result
            )
            assert result[0].node.get_content() == mock_nodes[0].node.get_content(
                metadata_mode=MetadataMode.NONE
            )


class TestCreateAgenticQueryEngine:
    @patch("components.agentic_retriever.agentic_retriever.Settings")
    @patch("components.agentic_retriever.agentic_retriever.LiteLLM")
    @patch("components.agentic_retriever.agentic_retriever.create_embedding_model")
    @patch("components.agentic_retriever.agentic_retriever.ChromaVectorStore")
    @patch("components.agentic_retriever.agentic_retriever.VectorStoreIndex")
    def test_agentic_mode_success(
        self,
        MockVectorStoreIndex,
        MockChromaVectorStore,
        MockCreateEmbeddingModel,
        MockLiteLLM,
        MockSettings,  # Patched Settings
        mock_config,
        mock_vector_store_instance,
    ):
        mock_config.retrieval.mode = "agentic"
        query_engine = create_agentic_query_engine(
            mock_config, mock_vector_store_instance
        )
        assert query_engine is not None
        MockLiteLLM.assert_called_once()
        # Assert that the global settings were configured
        assert MockSettings.llm is not None
        assert MockSettings.embed_model is not None

    # FIX: Patch llama_index.core.Settings to prevent global state leakage.
    @patch("components.agentic_retriever.agentic_retriever.Settings")
    @patch("components.agentic_retriever.agentic_retriever.create_embedding_model")
    @patch("components.agentic_retriever.agentic_retriever.ChromaVectorStore")
    @patch("components.agentic_retriever.agentic_retriever.VectorStoreIndex")
    def test_static_mode_success(
        self,
        MockVectorStoreIndex,
        MockChromaVectorStore,
        MockCreateEmbeddingModel,
        MockSettings,  # Patched Settings
        mock_config,
        mock_vector_store_instance,
    ):
        mock_config.retrieval.mode = "static"
        query_engine = create_agentic_query_engine(
            mock_config, mock_vector_store_instance
        )
        assert query_engine is not None
        assert isinstance(query_engine, ExpandedSourceQueryEngine)
        # Assert that the global settings were configured
        assert MockSettings.llm is not None
        assert MockSettings.embed_model is not None

    def test_agentic_mode_missing_generation_model(
        self, mock_config, mock_vector_store_instance
    ):
        mock_config.retrieval.mode = "agentic"
        mock_config.generation_model = None
        query_engine = create_agentic_query_engine(
            mock_config, mock_vector_store_instance
        )
        assert query_engine is None

    @patch(
        "components.agentic_retriever.agentic_retriever.create_embedding_model",
        side_effect=Exception("Embedding error"),
    )
    def test_create_query_engine_error_handling(
        self, MockCreateEmbeddingModel, mock_config, mock_vector_store_instance
    ):
        mock_config.retrieval.mode = "static"
        query_engine = create_agentic_query_engine(
            mock_config, mock_vector_store_instance
        )
        assert query_engine is None


class TestStaticContextPostprocessor:
    def test_init(self):
        postprocessor = StaticContextPostprocessor()
        assert postprocessor._document_reader is not None

    def test_postprocess_nodes_no_nodes(self):
        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes([])
        assert result == []

    def test_postprocess_nodes_no_query_bundle(self, mock_nodes):
        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes(mock_nodes, query_bundle=None)
        assert result == mock_nodes

    @patch(
        "components.document_processing.document_tools.DocumentReader.get_enclosing_sections"
    )
    def test_postprocess_nodes_success(self, mock_get_enclosing_sections):
        mock_get_enclosing_sections.return_value = ("Expanded content", 0, 100)
        node = NodeWithScore(
            node=TextNode(
                text="Original content",
                metadata={"file_path": "test.md"},
                start_char_idx=0,
                end_char_idx=10,
            ),
            score=0.5,
        )
        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes([node])
        assert len(result) == 1
        assert (
            result[0].node.get_content(metadata_mode=MetadataMode.NONE)
            == "Expanded content"
        )
        assert result[0].node.metadata["start_char_idx"] == 0
        assert result[0].node.metadata["end_char_idx"] == 100
        mock_get_enclosing_sections.assert_called_once_with("test.md", 0, 10)

    @patch(
        "components.document_processing.document_tools.DocumentReader.get_enclosing_sections",
        side_effect=Exception("Reader error"),
    )
    def test_postprocess_nodes_error_handling(self, mock_get_enclosing_sections):
        node = NodeWithScore(
            node=TextNode(
                text="Original content",
                metadata={"file_path": "test.md"},
                start_char_idx=0,
                end_char_idx=10,
            ),
            score=0.5,
        )
        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes([node])
        assert len(result) == 1
        assert (
            result[0].node.get_content(metadata_mode=MetadataMode.NONE)
            == "Original content"
        )

    def test_postprocess_nodes_deduplication(self):
        node1 = NodeWithScore(
            node=TextNode(
                text="Content A",
                metadata={"file_path": "file.md"},
                start_char_idx=0,
                end_char_idx=10,
            ),
            score=0.8,
        )
        node2 = NodeWithScore(
            node=TextNode(
                text="Content B",
                metadata={"file_path": "file.md"},
                start_char_idx=0,
                end_char_idx=10,
            ),
            score=0.7,
        )
        postprocessor = StaticContextPostprocessor()
        with patch(
            "components.document_processing.document_tools.DocumentReader.get_enclosing_sections",
            return_value=("Expanded Content", 0, 10),
        ):
            result = postprocessor._postprocess_nodes([node1, node2])
            assert len(result) == 1
            assert result[0].score == 0.8
            assert (
                result[0].node.get_content(metadata_mode=MetadataMode.NONE)
                == "Expanded Content"
            )

    def test_postprocess_nodes_deduplication_different_sections(self):
        node1 = NodeWithScore(
            node=TextNode(
                text="Content A",
                metadata={
                    "file_path": "file.md",
                    "start_char_idx": 0,
                    "end_char_idx": 10,
                },
            ),
            score=0.8,
        )
        node2 = NodeWithScore(
            node=TextNode(
                text="Content B",
                metadata={
                    "file_path": "file.md",
                    "start_char_idx": 20,
                    "end_char_idx": 30,
                },
            ),
            score=0.7,
        )
        postprocessor = StaticContextPostprocessor()
        with patch(
            "components.document_processing.document_tools.DocumentReader.get_enclosing_sections",
            side_effect=[("Expanded Content A", 0, 10), ("Expanded Content B", 20, 30)],
        ):
            result = postprocessor._postprocess_nodes([node1, node2])
            assert len(result) == 2

    def test_postprocess_nodes_missing_metadata(self):
        node = NodeWithScore(node=TextNode(text="Original content"), score=0.5)
        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes([node])
        assert len(result) == 1
        assert (
            result[0].node.get_content(metadata_mode=MetadataMode.NONE)
            == "Original content"
        )

    def test_postprocess_nodes_no_expansion(self):
        node = NodeWithScore(
            node=TextNode(
                text="Original content",
                metadata={"file_path": "test.md"},
                start_char_idx=0,
                end_char_idx=10,
            ),
            score=0.5,
        )
        postprocessor = StaticContextPostprocessor()
        with patch(
            "components.document_processing.document_tools.DocumentReader.get_enclosing_sections",
            return_value=("Original content", 0, 10),
        ):
            result = postprocessor._postprocess_nodes([node])
            assert len(result) == 1
            assert (
                result[0].node.get_content(metadata_mode=MetadataMode.NONE)
                == "Original content"
            )
