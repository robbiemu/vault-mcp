# Vault MCP Configuration Guide

This guide provides detailed information on configuring your Vault MCP server beyond the minimal setup described in the Quick Start guide.

## Configuration Files

Vault MCP uses two main configuration files:

- `config/app.toml` - Main application configuration
- `config/prompts.toml` - AI/LLM prompts for agentic processing

## Configuration Loading

The server loads configuration from `config/app.toml` by default. You can either:

1. **Edit the main config directly** (simplest approach):
   ```bash
   # Edit the main configuration file
   editor config/app.toml
   ```

2. **Create a custom config file** (for deployment scenarios):
   ```bash
   cp config/app.toml my-config.toml
   # Then run with custom config using CLI options
   vault-mcp --app-config my-config.toml
   ```

## Command-Line Configuration Options

### Server Control
- `--serve-api`: Run the standard API server (default: both servers)
- `--serve-mcp`: Run the MCP-compliant server (default: both servers)
- `--serve-mcp-stdio`: Run the MCP server over stdio for agent-managed launching
- `--api-port`: Port for the API server (default: 8000)
- `--mcp-port`: Port for the MCP server (default: 8081)
- `--host`: Override the server host (default: from config)

### Configuration Overrides
- `-c, --config`: Path to a directory containing `app.toml` and `prompts.toml`
- `-a, --app-config`: Path to a specific `app.toml` file
- `-p, --prompts-config`: Path to a specific `prompts.toml` file
- `--database-dir`: Override the vector database directory

### Examples
```bash
# Run with custom configuration
vault-mcp --app-config ./configs/project_a_config.toml

# Run only MCP server on custom port with custom database
vault-mcp --serve-mcp --mcp-port 9000 --database-dir ./custom_db

# Run MCP server over stdio for agents that manage the process
vault-mcp --serve-mcp-stdio --database-dir ./custom_db

# Run both servers with custom host
vault-mcp --host 0.0.0.0 --api-port 8080 --mcp-port 8081
```

## Configuration Sections

### `[paths]` - Document Sources

```toml
[paths]
# The directory where your documents are stored
vault_dir = "/path/to/your/documents"
# Document source type
type = "Obsidian"  # "Standard", "Obsidian", or "Joplin"
# Directory to store the ChromaDB vector database
database_dir = "./chroma_db"
```

**Document Source Types:**

#### Standard - Plain Markdown Files
```toml
[paths]
vault_dir = "/path/to/markdown/files"
type = "Standard"
```
For plain Markdown files in a directory structure.

#### Obsidian - Obsidian Vault
```toml
[paths]
vault_dir = "/path/to/obsidian/vault"
type = "Obsidian"
```
For Obsidian vaults with wiki-links and metadata support.

#### Joplin - Joplin Notebooks
```toml
[paths]
type = "Joplin"

[joplin_config]
api_token = "your-joplin-api-token"
```

**Joplin Setup:**
1. Enable Joplin Web Clipper: Tools → Options → Web Clipper
2. Copy the authorization token from Web Clipper settings
3. Add token to configuration

### `[server]` - Server Configuration

```toml
[server]
# Server host address
host = "127.0.0.1"
# API server port (when running both servers)
port = 8000
```

### `[retrieval]` - Processing Mode

```toml
[retrieval]
# Processing mode: "agentic" or "static"
mode = "static"
```

**Mode Options:**

#### Static Mode (Fast, Deterministic)
```toml
[retrieval]
mode = "static"
# No generation_model section needed
```
- Expands chunks to full section context
- No LLM calls required
- Fast and predictable

#### Agentic Mode (AI-Enhanced)
```toml
[retrieval]
mode = "agentic"

# Required for agentic mode
[generation_model]
model_name = "gpt-4o-mini"
[generation_model.parameters]
temperature = 0.3
max_tokens = 2000
```
- Uses LLM to rewrite and enhance chunks
- Higher quality, context-aware responses
- Requires generation model configuration

### `[prefix_filter]` - File Filtering

```toml
[prefix_filter]
# Only index files starting with these prefixes
# Empty list = include all .md files
allowed_prefixes = [
  "Project Documentation",
  "Meeting Notes",
  "Research"
]
```

### `[indexing]` - Document Processing

```toml
[indexing]
# Maximum size of text chunks (characters)
chunk_size = 1024
# Overlap between consecutive chunks (characters)
chunk_overlap = 200
# Minimum quality score for chunks (0.0-1.0)
quality_threshold = 0.6
```

### `[watcher]` - Live File Monitoring

```toml
[watcher]
# Enable/disable live file watching
enabled = true
# Delay before processing file changes (seconds)
debounce_seconds = 2.0
```

### `[embedding_model]` - Embedding Configuration

The server supports various embedding providers through a pluggable system:

#### Sentence Transformers (Default)
```toml
[embedding_model]
provider = "sentence_transformers"
model_name = "all-MiniLM-L6-v2"  # Any Hugging Face model
```

#### MLX Embeddings (Apple Silicon)
```toml
[embedding_model]
provider = "mlx_embeddings"
model_name = "mlx-community/mxbai-embed-large-v1"
```

#### OpenAI-Compatible API (Ollama, etc.)
```toml
[embedding_model]
provider = "openai_endpoint"
model_name = "nomic-embed-text"
endpoint_url = "http://localhost:11434/v1"
api_key = "ollama"  # Required, even if unused
```

#### Custom Plugin Wrappers
For advanced models requiring special formatting:

```toml
[embedding_model]
provider = "openai_endpoint"
model_name = "yoeven/multilingual-e5:large-it-Q5_K_M"
endpoint_url = "http://localhost:11434/v1"
api_key = "ollama"
# Custom wrapper for instruction-tuned models
wrapper_class = "plugins.e5_instruct_wrapper.E5InstructWrapper"
```

### `[generation_model]` - LLM Configuration

Required for agentic mode. Uses [LiteLLM](https://docs.litellm.ai/docs/providers) for unified model access:

#### Local Models (Ollama)
```toml
[generation_model]
model_name = "ollama/llama3"  # or ollama/mistral, ollama/codellama
[generation_model.parameters]
temperature = 0.5
max_tokens = 1024
```

#### OpenAI Models
```toml
[generation_model]
model_name = "gpt-4o-mini"
[generation_model.parameters]
temperature = 0.3
max_tokens = 2000
```

#### Anthropic Models
```toml
[generation_model]
model_name = "claude-3-haiku-20240307"
[generation_model.parameters]
temperature = 0.4
max_tokens = 1500
```

### `[joplin_config]` - Joplin Integration

```toml
[joplin_config]
# API token from Joplin Web Clipper settings
api_token = "your-api-token-here"
```

## Environment Variables

For cloud-based models, set appropriate environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Azure OpenAI
export AZURE_API_KEY="your-azure-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"

# Logging level
export LOG_LEVEL=DEBUG  # For troubleshooting
```

## Configuration Examples

### Minimal Setup (Static Mode)
```toml
[paths]
vault_dir = "/Users/me/Documents/notes"
type = "Obsidian"

[retrieval]
mode = "static"

[prefix_filter]
allowed_prefixes = []  # Index all files
```

### Production Setup (Agentic Mode)
```toml
[paths]
vault_dir = "/data/knowledge-base"
type = "Standard"
database_dir = "/data/chroma_db"

[server]
host = "0.0.0.0"
port = 8000

[retrieval]
mode = "agentic"

[generation_model]
model_name = "gpt-4o-mini"
[generation_model.parameters]
temperature = 0.3
max_tokens = 2000

[embedding_model]
provider = "openai_endpoint"
model_name = "nomic-embed-text"
endpoint_url = "http://localhost:11434/v1"
api_key = "ollama"

[prefix_filter]
allowed_prefixes = [
  "Documentation",
  "Procedures",
  "Reference"
]

[indexing]
chunk_size = 1024
chunk_overlap = 200
quality_threshold = 0.7

[watcher]
enabled = true
debounce_seconds = 3.0
```

### Joplin Integration
```toml
[paths]
type = "Joplin"

[joplin_config]
api_token = "abc123def456..."

[retrieval]
mode = "agentic"

[generation_model]
model_name = "claude-3-haiku-20240307"
[generation_model.parameters]
temperature = 0.4
max_tokens = 1500
```

## Troubleshooting Configuration

### Server Won't Start
- Verify `vault_dir` exists and is accessible
- Check TOML syntax with a validator
- Ensure required environment variables are set
- Check file permissions

### Files Not Being Indexed
- Verify `allowed_prefixes` configuration
- Check file permissions in vault directory
- Review server logs for errors
- Test with an empty `allowed_prefixes` array

### Search Returns No Results
- Lower the `quality_threshold` value
- Check if files were indexed (`GET /files`)
- Verify search queries are relevant
- Review chunk processing settings

### Live Sync Not Working
- Ensure `watcher.enabled = true`
- Check vault directory permissions
- Review file watcher logs
- Verify debounce settings

### Model Configuration Issues
- Verify environment variables are set correctly
- Test model connectivity independently
- Check LiteLLM compatibility for your provider
- Review embedding model requirements

## Applying Configuration Changes

After modifying configuration files:

1. **Restart the server** for changes to take effect
2. **Force reindexing** if document processing settings changed:
   ```bash
   curl -X POST http://localhost:8000/reindex
   ```
3. **Check logs** for configuration validation errors
4. **Test functionality** with a simple query

## Security Considerations

- Never commit API keys to version control
- Use environment variables for sensitive data
- Restrict server host binding in production
- Consider firewall rules for exposed services
- Use HTTPS in production deployments
