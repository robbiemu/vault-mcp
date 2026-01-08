"""Tests for Vector Store dimension migration."""

import pytest
from unittest.mock import MagicMock, patch
from components.vector_store.vector_store import VectorStore
from shared.config import EmbeddingModelConfig

class MockEmbeddingModel:
    def __init__(self, dimension):
        self.dimension = dimension

    def encode(self, texts):
        return [[0.1] * self.dimension for _ in texts]

@pytest.fixture
def mock_embedding_factory():
    with patch("components.vector_store.vector_store.create_embedding_model") as mock:
        yield mock

def test_vector_store_dimension_mismatch_reset(tmp_path, mock_embedding_factory):
    """Test that vector store resets when embedding dimension changes."""
    persist_dir = str(tmp_path / "test_migration")
    
    # 1. Initialize with dimension 10
    mock_embedding_factory.return_value = MockEmbeddingModel(dimension=10)
    
    config1 = EmbeddingModelConfig(provider="test", model_name="model-10")
    vs1 = VectorStore(
        embedding_config=config1,
        persist_directory=persist_dir,
        collection_name="test_coll"
    )
    
    # Add some data
    vs1.add_chunks([{
        "text": "test",
        "file_path": "test.md",
        "chunk_id": "1",
        "score": 1.0
    }])
    assert vs1.get_chunk_count() == 1
    
    # 2. Re-initialize with dimension 20 on the same directory
    # The VectorStore __init__ logic should detect the mismatch
    mock_embedding_factory.return_value = MockEmbeddingModel(dimension=20)
    
    config2 = EmbeddingModelConfig(provider="test", model_name="model-20")
    vs2 = VectorStore(
        embedding_config=config2,
        persist_directory=persist_dir,
        collection_name="test_coll"
    )
    
    # The collection should have been recreated, so count should be 0
    assert vs2.get_chunk_count() == 0
    
    # Verify we can add new chunks with new dimension
    vs2.add_chunks([{
        "text": "test2",
        "file_path": "test2.md",
        "chunk_id": "2",
        "score": 1.0
    }])
    assert vs2.get_chunk_count() == 1
