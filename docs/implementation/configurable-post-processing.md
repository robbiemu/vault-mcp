# Configurable Post-Processing Feature

## Overview

The configurable post-processing feature provides flexibility for processing Retrieval-Augmented Generation (RAG) queries with two modes:

1. **Agentic Mode**: Uses LLM-powered agents for rewriting and enhancing retrieved content to produce high-quality, context-aware responses.
2. **Static Mode**: Expands content to its full section context using non-LLM deterministic methods, suitable for faster operations.

This allows users to balance between performance and response quality for different use cases.

## Configuration

### Mode Selection
- **`RetrievalConfig`**
  - `mode: str` - Either "agentic" or "static"

### Post-Processing Modes

#### Static Mode
- Expands chunks to full section context using existing document structure
- Benefits: Fast, no LLM required

#### Agentic Mode
- Enhanced rewriting with LLM for improved relevance and coherence
- Requires a configured generation model

## Key Interfaces and Usage

### Configuration Example

#### Static Mode
```toml
[retrieval]
mode = "static"

# LLM generation_model is OPTIONAL for static mode
```

#### Agentic Mode
```toml
[retrieval]
mode = "agentic"

generation_model = "required"
```

### Creating a Query Engine
Using the configured mode, create a query engine to process RAG queries:

```python
from vault_mcp.config import Config
from components.agentic_retriever import create_agentic_query_engine

# Configure retrieval mode
config = Config(
    retrieval = {'mode': 'static'},
    # Optional for static mode
    generation_model = None 
)

# Initialize query engine
query_engine = create_agentic_query_engine(config, vector_store)

# Execute query
response = query_engine.query("What are the system capabilities?")
```

## Testing and Validation

### Unit Tests

- **Static Mode Tests**
  - Tests expansion and deduplication logic

- **Agentic Mode Tests**
  - Validates generation model requirement

### Integration Testing

Both modes are exercised through the query interface to ensure seamless switching and consistent behavior.

## Future Enhancements

1. **Hybrid Mode**: Possibility of combining static expansion with minor LLM enhancements.
2. **Performance Metrics**: Implement internal metrics for comparing mode costs and benefits.
3. **Enhanced Cache**: Utilize caching mechanisms to further boost static mode performance.
