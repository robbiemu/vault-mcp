# Vector Store Component

## Purpose and Scope

The Vector Store component manages document embeddings and provides semantic search capabilities. It handles the storage of document chunks as vector embeddings using ChromaDB and enables similarity-based retrieval for the MCP server's query functionality.

## Key Interfaces and APIs

### Core Classes

- **`VectorStore`** - Main class for vector operations
  - `add_chunks(chunks)` - Add document chunks to the vector store
  - `search(query, limit, quality_threshold)` - Semantic search for chunks
  - `remove_file_chunks(file_path)` - Remove chunks from a specific file
  - `get_all_file_paths()` - List all indexed file paths
  - `clear_all()` - Clear all stored data

### Data Models

- **`ChunkMetadata`** - Represents a document chunk with metadata
  - `text` - The chunk content
  - `file_path` - Source file path
  - `chunk_id` - Unique chunk identifier
  - `score` - Quality score of the chunk

## Dependencies and Integration Points

### Internal Dependencies
- **MCP Server models** - For ChunkMetadata data structure

### External Dependencies
- **ChromaDB** - Vector database for embedding storage
- **Sentence Transformers** - Text embedding generation (all-MiniLM-L6-v2 model)

### Integration Points
- **File Watcher component** - Receives chunk updates from file changes
- **MCP Server component** - Provides search results for API queries
- **Document processing** - Stores processed chunks from documents

## Configuration Requirements

Configuration is handled internally with reasonable defaults, but the `VectorStore` now requires an `embedding_config` object during initialization:

```python
from vault_mcp.config import EmbeddingModelConfig

# Example configuration (this would typically come from app.toml)
embedding_config = EmbeddingModelConfig(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2"
)

VectorStore(
    embedding_config=embedding_config,
    persist_directory="./chroma_db",  # Data persistence location
    collection_name="vault_docs"      # ChromaDB collection name
)
```

Quality filtering uses shared configuration:
```toml
[indexing]
quality_threshold = 0.75  # Minimum score for search results
```

## Usage Examples

### Basic Operations
```python
from components.vector_store.vector_store import VectorStore
from vault_mcp.config import EmbeddingModelConfig

# Initialize vector store with an embedding configuration
embedding_config = EmbeddingModelConfig(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2"
)
vector_store = VectorStore(embedding_config=embedding_config)

# Add document chunks
chunks = [{
    "text": "This is a document chunk about authentication.",
    "file_path": "auth.md",
    "chunk_id": "auth.md|0",
    "score": 0.85
}]
vector_store.add_chunks(chunks)

# Search for relevant chunks
results = vector_store.search(
    query="authentication system",
    limit=5,
    quality_threshold=0.7
)

# Remove chunks from a file
vector_store.remove_file_chunks("auth.md")
```

### Integration with File Changes
```python
# When a file is modified
vector_store.remove_file_chunks(file_path)  # Remove old chunks
vector_store.add_chunks(new_chunks)         # Add updated chunks

# When a file is deleted
vector_store.remove_file_chunks(file_path)  # Remove all chunks
```
