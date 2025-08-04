# Configurable Post-Processing Component

## Purpose and Scope

The Configurable Post-Processing component provides runtime flexibility for RAG (Retrieval-Augmented Generation) queries by offering two distinct post-processing modes:

1. **Agentic Mode**: Uses LLM-powered agents to rewrite and enhance retrieved chunks for high-quality, context-aware responses
2. **Static Mode**: Expands retrieved chunks to their full section context using deterministic text processing (fast, no LLM required)

This allows users to choose between computational performance and answer quality based on their specific use case requirements.

## Key Interfaces and APIs

### Configuration

- **`RetrievalConfig`** - Configuration model for retrieval post-processing
  - `mode: str` - Either "agentic" or "static" (default: "agentic")

### Post-Processors

- **`StaticContextPostprocessor`** - Expands chunks to full section context
  - `_postprocess_nodes(nodes, query_bundle) -> List[NodeWithScore]` - Core processing logic
  - Automatically deduplicates identical sections
  - Uses existing `DocumentReader.get_enclosing_sections()` method

- **`ChunkRewriterPostprocessor`** - LLM-powered chunk rewriting (existing)
  - Requires generation model configuration
  - Provides enhanced context through agent-driven rewriting

### Factory Function

- **`create_agentic_query_engine(config, vector_store, ...)` -> BaseQueryEngine**
  - Factory that selects appropriate post-processor based on `config.retrieval.mode`
  - Validates configuration requirements for selected mode
  - Returns configured query engine ready for RAG operations

## Dependencies and Integration Points

### Internal Dependencies
- **Config module** - Uses `RetrievalConfig` and validates `GenerationModelConfig` requirements
- **Document Processing** - Uses `DocumentReader.get_enclosing_sections()` for static mode
- **Agentic Retriever** - Existing `ChunkRewriterPostprocessor` for agentic mode

### External Dependencies
- **LlamaIndex Core** - `BaseNodePostprocessor`, `NodeWithScore`, `QueryBundle`, `TextNode`
- **LiteLLM** - For agentic mode LLM operations (only when required)

### Integration Points
- **MCP Server** - Query engine creation and `/mcp/query` endpoint processing
- **Vector Store** - Retrieval of initial document chunks before post-processing

## Configuration Requirements

### Static Mode Configuration

```toml
[retrieval]
mode = "static"

# generation_model section is OPTIONAL for static mode
# The server will start successfully without any LLM configuration
```

### Agentic Mode Configuration

```toml
[retrieval]
mode = "agentic"

[generation_model]  # REQUIRED for agentic mode
model_name = "gpt-4o-mini"
[generation_model.parameters]
temperature = 0.3
max_tokens = 2000
```

### Validation Rules

1. **Agentic Mode**: `generation_model` configuration is mandatory
   - Server will fail at startup if `generation_model` is missing
   - Clear error message guides user to provide required configuration

2. **Static Mode**: `generation_model` configuration is optional
   - Server starts successfully even if `[generation_model]` section is commented out or deleted
   - No LLM dependencies required

## Usage Examples

### Static Mode Usage (Fast, Deterministic)

```python
from vault_mcp.config import Config, RetrievalConfig
from components.agentic_retriever import create_agentic_query_engine

# Configure for static mode
config = Config(
    # ... other config ...
    retrieval=RetrievalConfig(mode="static"),
    generation_model=None  # Not required
)

# Create query engine
query_engine = create_agentic_query_engine(config, vector_store)

# Execute query - returns expanded sections
response = query_engine.query("How does the system work?")
```

### Agentic Mode Usage (High Quality, Slower)

```python
from vault_mcp.config import Config, RetrievalConfig, GenerationModelConfig

# Configure for agentic mode
config = Config(
    # ... other config ...
    retrieval=RetrievalConfig(mode="agentic"),
    generation_model=GenerationModelConfig(
        model_name="gpt-4o-mini",
        parameters={"temperature": 0.3, "max_tokens": 2000}
    )
)

# Create query engine
query_engine = create_agentic_query_engine(config, vector_store)

# Execute query - returns LLM-enhanced responses
response = query_engine.query("How does the system work?")
```

### Configuration Switching

```bash
# Run with static mode configuration
vault-mcp --app-config config/examples/static_mode_example.toml

# Run with agentic mode configuration  
vault-mcp --app-config config/examples/agentic_mode_example.toml
```

## Implementation Details

### Static Mode Processing

1. **Chunk Expansion**: Each retrieved chunk is expanded to its full enclosing section(s)
2. **Section Detection**: Uses existing markdown header parsing to identify section boundaries
3. **Deduplication**: Multiple chunks from the same section are merged into a single result
4. **Score Preservation**: Maintains the highest relevance score among deduplicated chunks

### Deduplication Algorithm

The static processor prevents duplicate sections using a content-based approach:

```python
# Create unique key based on file path and section content
section_key = f"{file_path}:{hash(section_content)}"

# Track seen sections and keep highest scoring version
if section_key not in seen_sections:
    seen_sections[section_key] = expanded_node
else:
    # Keep higher scoring node
    if node.score > existing_node.score:
        seen_sections[section_key] = new_expanded_node
```

### Performance Characteristics

| Mode | Response Time | Quality | LLM Required | Use Cases |
|------|---------------|---------|--------------|-----------|
| Static | Very Fast (~100ms) | Good | No | Debugging, development, performance-critical |
| Agentic | Slower (~2-5s) | Excellent | Yes | Production, complex queries, synthesis |

### Error Handling

- **Missing Metadata**: Nodes without required metadata (`file_path`, character indices) are kept unchanged
- **File Access Errors**: Gracefully handled with logging, original chunks returned
- **Invalid Mode**: Clear validation error with supported mode list
- **Missing LLM Config**: Startup failure with actionable error message for agentic mode

## Testing

### Unit Tests

- `test_static_postprocessor.py` - Tests for static mode functionality
  - Basic expansion and deduplication
  - Missing metadata handling
  - Empty input handling

- `test_retrieval_config.py` - Tests for configuration validation
  - Agentic mode LLM requirement validation
  - Static mode optional LLM configuration
  - Default behavior verification

### Integration Testing

Both modes are tested through the same query interface, ensuring consistent behavior and seamless switching between modes.

## Future Enhancements

1. **Hybrid Mode**: Combine static expansion with lightweight LLM enhancement
2. **Caching**: Cache expanded sections to improve static mode performance further
3. **Configurable Deduplication**: Allow users to control deduplication behavior
4. **Performance Metrics**: Built-in timing and quality metrics for mode comparison
