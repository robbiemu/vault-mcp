# 🧾 Vault MCP - Obsidian Documentation Server

**Version 0.2.3**

A **Model Context Protocol (MCP)** compliant server that indexes, searches, and serves Obsidian vault documents with **semantic search (RAG)**, **live synchronization**, and **quality-based chunk filtering**.

## ✨ Features

- 🤖 **Retrieval-Augmented Generation (RAG)**: Enhanced document retrieval with text generation for comprehensive answers using context-aware AI models
- 🔍 **Semantic Search**: Vector-based search across your Obsidian vault
- 📁 **Prefix Filtering**: Only index files matching specific filename prefixes
- 🔄 **Live Sync**: Automatically re-indexes files when they change on disk
- 📊 **Quality Scoring**: Filters document chunks based on content quality
- 🔌 **MCP Compliant**: Follows Model Context Protocol standards
- 🚀 **FastAPI Backend**: RESTful API with automatic documentation
- 📝 **Markdown Processing**: Obsidian-friendly markdown parsing

## 📚 Table of Contents

- [🏗️ Architecture](#%EF%B8%8F-architecture)
- [🚀 Quick Start](#-quick-start)
- [📋 API Endpoints](#-api-endpoints)
- [⚙️ Configuration Options](#%EF%B8%8F-configuration-options)
- [🔧 Development](#-development)
- [🧠 How It Works](#-how-it-works)
- [🎯 Use Cases](#-use-cases)
- [🔍 Quality Scoring](#-quality-scoring)
- [🚨 Troubleshooting](#-troubleshooting)
- [📊 Performance](#-performance)
- [🤝 Contributing](#-contributing)

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "User Environment"
        OV["📁 Obsidian Vault<br/>(.md files)"]
        AI["🤖 AI Agent<br/>(MCP Client)"]
    end
    
    subgraph "Vault MCP Server"
        subgraph "Components"
            MS["🖥️ MCP Server<br/>(FastAPI)"]
            VS["🔍 Vector Store<br/>(ChromaDB)"]
            FW["👁️ File Watcher<br/>(Live Sync)"]
            AR["🤖 Agentic Retriever<br/>(LlamaIndex)"]
        end
        
        subgraph "Shared Utilities"
            DP["📄 Document Processor"]
            CF["⚙️ Configuration"]
        end
    end
    
    subgraph "External Services"
        LLM["🚀 LLM Providers<br/>(OpenAI/Anthropic/Ollama)"]
    end
    
    %% Data flow connections
    OV -->|"Monitor .md files"| FW
    FW -->|"Process changes"| DP
    DP -->|"Generate chunks"| VS
    AI <-->|"HTTP/REST API (MCP Protocol)"| MS
    MS -->|"Search queries"| VS
    MS -->|"RAG queries"| AR
    MS -->|"Document retrieval"| OV
    AR <-->|"Generate answers"| LLM
    
    %% Configuration connections
    CF -.->|"Configure"| MS
    CF -.->|"Configure"| FW
    CF -.->|"Configure"| VS
    CF -.->|"Configure"| AR
    
    %% Styling
    classDef component fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef external fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef utility fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class MS,VS,FW,AR component
    class OV,AI,LLM external
    class DP,CF utility
```

## 🚀 Quick Start

### Prerequisites

This project uses **[uv](https://docs.astral.sh/uv/)** for fast, reliable Python package management. Install uv first:

```bash
# Install uv (recommended method)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv

# Or via Homebrew (macOS)
brew install uv
```

### 1. Installation

Clone the repository and set up with `uv`:

```bash
git clone <repository-url>
cd vault-mcp

# Create virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Install dependencies with uv
uv sync
```

For development with additional tools:

```bash
# Install with development dependencies
uv sync --extra dev
```

### 2. Configuration

The server loads configuration from `config/app.toml`. You can either:

1. **Edit the main config directly** (simplest approach):
   ```bash
   # Edit the main configuration file
   editor config/app.toml
   ```

2. **Create a custom config file** (for deployment scenarios):
   ```bash
   cp config/app.toml my-config.toml
   # Then run with: uvicorn components.mcp_server.main:app --reload --env-file my-config.toml
   ```

Here are the most common settings to customize:

```toml
# Update the vault path to point to your Obsidian vault
[paths]
vault_dir = "/path/to/your/obsidian-vault"

# Override the generation model if desired
[generation_model]
model_name = "gpt-4o-mini"  # or "claude-3-haiku-20240307", "ollama/mistral", etc.
[generation_model.parameters]
temperature = 0.3
max_tokens = 2000

# Customize file filtering if needed
[prefix_filter]
allowed_prefixes = [
  "Project Documentation",
  "Meeting Notes",
  "Research"
]
```

The base configuration file (`config/app.toml`) contains comprehensive examples and documentation for all available options.

### 3. Run the Server

```bash
# Using the installed script
vault-mcp
```

The server will start and automatically index your vault. Visit `http://localhost:8000/docs` for the interactive API documentation.

#### Command-Line Overrides

You can override the default configuration file locations using command-line flags. This is useful for running multiple server instances against different parts of your vault.

- `-c, --config`: Path to a directory containing `app.toml` and `prompts.toml`.
- `-a, --app-config`: Path to a specific `app.toml` file.
- `-p, --prompts-config`: Path to a specific `prompts.toml` file.

**Example:**

```bash
# Run the server using a custom app config for a specific project
vault-mcp --app-config ./configs/project_a_config.toml

# Run with a completely different configuration set
vault-mcp --config ./project_b_config/
```

## 📋 API Endpoints

### GET `/mcp/info`
Get server capabilities and configuration.

**Response (example with RAG capabilities):**
```json
{
  "mcp_version": "1.0",
  "capabilities": ["search", "document_retrieval", "live_sync", "introspection", "rag_generation"],
  "indexed_files": ["Resource Balance Game - Goals.md", "..."],
  "config": {
    "chunk_size": 512,
    "overlap": 64,
    "quality_threshold": 0.75
  }
}
```

### GET `/mcp/files`
List all indexed files.

**Response:**
```json
{
  "files": ["Resource Balance Game - Goals.md", "..."],
  "total_count": 15
}
```

### GET `/mcp/document?file_path=...`
Retrieve full document content.

**Parameters:**
- `file_path`: Path to the document file

**Response:**
```json
{
  "content": "# Document Title\n\nDocument content...",
  "file_path": "/path/to/document.md",
  "metadata": null
}
```

### POST `/mcp/query`
Perform an agentic retrieval-augmented generation query across indexed documents.

**Request:**
```json
{
  "query": "How does the game economy work?",
  "limit": 5
}
```

**Response:**
```json
{
  "sources": [
    {
      "text": "Economy chunk content...",
      "file_path": "Resource Balance Game - Economy.md",
      "chunk_id": "file|0",
      "score": 0.88
    }
  ]
}
```

### POST `/mcp/reindex`
Force a full re-index of the vault.

**Response:**
```json
{
  "success": true,
  "message": "Reindexing completed",
  "files_processed": 15
}
```

## ⚙️ Configuration Options

### Paths
- `vault_dir`: Absolute path to your Obsidian vault directory

### Prefix Filter
- `allowed_prefixes`: List of filename prefixes to include in indexing
- Empty list = include all `.md` files

### Indexing
- `chunk_size`: Maximum size of text chunks (characters)
- `chunk_overlap`: Overlap between consecutive chunks (characters)
- `quality_threshold`: Minimum quality score for chunks (0.0-1.0)

### Watcher
- `enabled`: Enable/disable live file watching
- `debounce_seconds`: Delay before processing file changes

### Server
- `host`: Server host address
- `port`: Server port number

### Model Configuration

#### Embedding Models

The server supports three embedding providers:

**1. Sentence Transformers (Default)**
```toml
[embedding_model]
provider = "sentence_transformers"
model_name = "all-MiniLM-L6-v2"  # or any Hugging Face model
```

**2. MLX Embeddings (Apple Silicon)**
```toml
[embedding_model]
provider = "mlx_embeddings"
model_name = "mlx-community/mxbai-embed-large-v1"
```

**3. OpenAI-Compatible API**
```toml
[embedding_model]
provider = "openai_endpoint"
model_name = "nomic-embed-text"
endpoint_url = "http://localhost:11434/v1"  # Ollama example
api_key = "ollama"  # Required, even if unused
```

#### Generation Models

The server uses [LiteLLM](https://github.com/BerriAI/litellm) for unified model access:

**Local Models (Ollama)**
```toml
[generation_model]
model_name = "ollama/llama3"  # or ollama/mistral, ollama/codellama
[generation_model.parameters]
temperature = 0.5
max_tokens = 1024
```

**OpenAI Models**
```toml
[generation_model]
model_name = "gpt-4o-mini"  # Set OPENAI_API_KEY environment variable
[generation_model.parameters]
temperature = 0.3
max_tokens = 2000
```

**Anthropic Models**
```toml
[generation_model]
model_name = "claude-3-haiku-20240307"  # Set ANTHROPIC_API_KEY environment variable
[generation_model.parameters]
temperature = 0.4
max_tokens = 1500
```

**Other Providers**
LiteLLM supports 100+ models. See the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for the complete list.

#### Environment Variables for API Keys

For cloud-based models, set the appropriate environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Azure OpenAI
export AZURE_API_KEY="your-azure-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"

# Add more as needed for other providers
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone and set up with uv
git clone <repository-url>
cd vault-mcp

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Unix/macOS

# Install with development dependencies
uv sync --extra dev

# Run tests
pytest

# Format code
black components/ vault_mcp/
ruff check --fix

# Type checking
mypy components/ vault_mcp/

# Security scanning
bandit -c pyproject.toml -r components/ vault_mcp/
```

### Project Structure

```
vault-mcp/
├── components/              # Discrete system components
│   ├── mcp_server/          # HTTP/REST API server component
│   │   ├── main.py          # FastAPI server and endpoints
│   │   ├── models.py        # API data models
│   │   └── tests/           # Server-specific tests
│   ├── vector_store/        # Document embedding and search component
│   │   ├── vector_store.py  # ChromaDB management
│   │   └── tests/           # Vector store tests
│   ├── file_watcher/        # Live file monitoring component
│   │   ├── file_watcher.py  # File system event handling
│   │   └── tests/           # File watcher tests
│   └── agentic_retriever/  # Agentic retrieval and RAG generation component
│       ├── agentic_retriever.py # Core retriever logic
│       └── tests/           # Agent-specific tests
├── vault_mcp/               # Shared utilities and libraries
│   ├── config.py            # Configuration management
│   ├── document_processor.py # Markdown processing and chunking
│   └── __init__.py
├── tests/                   # Root-level integration tests
├── docs/
│   ├── components/          # Component-specific documentation
│   └── plan.md              # Technical specification
├── config/
│   ├── app.toml             # Main application configuration
│   └── prompts.toml         # AI/LLM prompts
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file
```

## 🧠 How It Works

1. **Document Ingestion**: The server scans your Obsidian vault for `.md` files matching the configured prefixes.

2. **Markdown Processing**: Files are parsed to extract plain text while maintaining structure.

3. **Chunking and Quality Scoring**: Text is broken into chunks, scored for quality to filter valuable content.

4. **Embedding and Storage**: High-quality chunks are embedded using sentence transformers and stored in ChromaDB.

5. **Agentic Retrieval**: The `Agentic Retriever` conducts agent-driven retrieval and refinement:
   - Searches relevant document chunks using `DocumentRAGTool`
   - Refines informative content via `ChunkRewriteAgent`
   - Manages concurrent processing with `ChunkRewriterPostprocessor`

6. **Live Sync and Query Handling**: A file watcher monitors the vault for changes, providing up-to-date search capabilities.

7. **Comprehensive Query Processing**: Queries are processed to deliver contextually rich and precise answers through a unified RAG pipeline.

## 🎯 Use Cases

- **AI-Powered Documentation**: Enable AI agents to search and reference your project documentation
- **Knowledge Base Search**: Semantic search across your personal knowledge vault
- **Research Assistant**: Quick retrieval of relevant information from large document collections
- **Documentation Sync**: Keep your AI tools synchronized with your latest documentation

## 🔍 Quality Scoring

The server uses several heuristics to score chunk quality:

- **Length Score**: Prefers medium-length chunks (100-800 characters)
- **Completeness**: Rewards chunks with complete sentences
- **Structure**: Favors chunks with headings, lists, or other structural elements
- **Content Richness**: Avoids chunks with excessive punctuation or very short words
- **Coherence**: Rewards good word diversity and topic focus

## 🚨 Troubleshooting

### Common Issues

**Server won't start:**
- Check that the vault directory exists and is accessible
- Verify configuration file syntax
- Ensure all dependencies are installed

**Files not being indexed:**
- Check the `allowed_prefixes` configuration
- Verify file permissions in the vault directory
- Look for error messages in the server logs

**Search returns no results:**
- Try lowering the `quality_threshold`
- Check if files were actually indexed (`/mcp/files`)
- Verify the search query is relevant to your content

**Live sync not working:**
- Ensure `watcher.enabled = true` in configuration
- Check that the vault directory is being monitored
- Look for file watcher error messages in logs

### Logging

The server uses Python's standard logging. To increase verbosity:

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG
uvicorn vault_mcp.main:app --reload
```

## 📊 Performance

- **Dataset Size**: Optimized for <24 files, ~50k tokens
- **Memory Usage**: Embeddings and vector store fit comfortably in RAM
- **Startup Time**: ~1-2 seconds for cold start
- **Search Latency**: Sub-second semantic search responses
- **File Watching**: Low CPU usage due to debouncing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [ChromaDB](https://www.trychroma.com/) for vector storage and retrieval
- [Sentence Transformers](https://www.sbert.net/) for embedding generation
- [Watchdog](https://github.com/gorakhargosh/watchdog) for file system monitoring
- [Mistune](https://mistune.lepture.com/) for markdown parsing

---

**Happy documenting! 📚✨**
