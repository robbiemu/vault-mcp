# Document Processing Component

## Purpose and Scope

The Document Processing component handles all aspects of document ingestion, content analysis, and reading operations. It provides a unified interface for loading documents from multiple sources (Standard, Obsidian, Joplin), assessing content quality, and extracting specific document sections.

## Key Interfaces and APIs

### Document Loading

- **`create_reader(config: Config) -> BaseReader`** - Factory function to create appropriate document readers
- **`load_documents(config: Config) -> List[Document]`** - Load and filter documents using configured reader
- **`DocumentLoaderError`** - Exception raised for document loading failures

### Quality Assessment

- **`ChunkQualityScorer`** - Content-based quality scoring for text chunks
  - `score(text: str) -> float` - Calculate quality score (0.0-1.0) based on:
    - Optimal Length (0.4 points): Prefers information-rich but not excessive length
    - Content Richness (0.3 points): Rewards substantial vocabulary
    - Information Density (0.3 points): Rewards diversity of meaningful words

### Document Reading

- **`DocumentReader`** - Core document reading functionality
  - `read_full_document(file_path: str) -> str` - Read entire document content
  - `get_enclosing_sections(file_path: str, start_char_idx: int, end_char_idx: int) -> str` - Extract sections containing character range

- **`FullDocumentRetrievalTool`** - Tool wrapper for full document retrieval
- **`SectionRetrievalTool`** - Tool wrapper for section-based retrieval

## Dependencies and Integration Points

### Internal Dependencies
- **Config module** - Uses reader type, vault paths, and filtering configuration
- **LlamaIndex readers** - ObsidianReader, JoplinReader, SimpleDirectoryReader

### External Dependencies
- **LlamaIndex Core** - Document schemas and reader interfaces
- **pathlib** - File system operations

### Integration Points
- **MCP Server** - Initial document indexing and API endpoints
- **File Watcher** - Live document processing and quality filtering
- **Agentic Retriever** - Document section retrieval for RAG queries
- **Vector Store** - Document chunk storage and retrieval

## Configuration Requirements

Uses shared configuration from `config/app.toml`:

```toml
[paths]
vault_dir = "/path/to/vault"  # Document source directory
type = "Standard"             # Reader type: Standard, Obsidian, Joplin

[joplin_config]
api_token = "your_token"      # Required for Joplin reader type

[prefix_filter]
allowed_prefixes = ["Doc"]    # File prefix filtering (Standard/Obsidian only)

[indexing]
quality_threshold = 0.75      # Minimum quality score for chunk inclusion
enable_quality_filter = true # Enable/disable quality filtering
```

## Usage Examples

### Multi-Source Document Loading
```python
from components.document_processing import load_documents, DocumentLoaderError
from vault_mcp.config import load_config

config = load_config()

try:
    documents = load_documents(config)
    print(f"Loaded {len(documents)} documents")
except DocumentLoaderError as e:
    print(f"Loading failed: {e}")
```

### Content Quality Assessment
```python
from components.document_processing import ChunkQualityScorer

scorer = ChunkQualityScorer()

# Assess chunk quality
chunk_text = "Advanced document processing systems implement sophisticated algorithms..."
quality_score = scorer.score(chunk_text)

if quality_score >= 0.75:
    print(f"High-quality chunk (score: {quality_score:.2f})")
```

### Document Section Retrieval
```python
from components.document_processing import DocumentReader

reader = DocumentReader()

# Get sections containing specific character range
sections = reader.get_enclosing_sections(
    file_path="/path/to/document.md",
    start_char_idx=100,
    end_char_idx=500
)
print(f"Enclosing sections: {sections}")
```

### Reader Type Configuration
```python
from components.document_processing import create_reader
from vault_mcp.config import Config, PathsConfig

# Obsidian vault
config = Config(paths=PathsConfig(vault_dir="/path/to/vault", type="Obsidian"))
reader = create_reader(config)

# Standard markdown directory
config.paths.type = "Standard"
reader = create_reader(config)

# Joplin notes (requires API token)
config.paths.type = "Joplin"
config.joplin_config.api_token = "your_token_here"
reader = create_reader(config)
```

## Quality Scoring Algorithm

The ChunkQualityScorer uses a content-focused approach with three weighted factors:

1. **Optimal Length (40% weight)**:
   - 150-1024 characters: 0.4 points (ideal information density)
   - 50-149 characters: 0.2 points (decent but brief)
   - Other lengths: 0.1 points (acceptable but not optimal)

2. **Content Richness (30% weight)**:
   - Average word length > 4.5: 0.3 points (sophisticated vocabulary)
   - Average word length > 3.5: 0.15 points (reasonable vocabulary)
   - Lower averages: 0 points

3. **Information Density (30% weight)**:
   - Unique word ratio > 0.6: 0.3 points (high diversity)
   - Unique word ratio > 0.4: 0.15 points (moderate diversity)
   - Lower ratios: 0 points

Scores are capped at 1.0 and chunks with < 3 words receive 0.0.
