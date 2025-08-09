# MCP Server Component

## Purpose and Scope

The MCP Server component provides the HTTP/REST API interface for the vault-mcp system. It implements the Model Context Protocol (MCP) endpoints and manages the server lifecycle, including startup initialization and graceful shutdown. This component orchestrates interactions between the file watcher and vector store components.

## Key Interfaces and APIs

### REST API Endpoints

- **`GET /mcp/info`** - Server introspection and capabilities
- **`GET /mcp/files`** - List indexed files
- **`GET /mcp/document`** - Retrieve full document content by file path
- **`POST /mcp/query`** - Semantic search across indexed documents
- **`POST /mcp/reindex`** - Force full vault re-indexing

### Core Classes

- **`FastAPI` app** - Main web application instance
- **Request/Response models** - Pydantic models for API data validation
- **Lifespan management** - Handles startup and shutdown events

## Dependencies and Integration Points

### Internal Dependencies
- **Vector Store component** - For document storage and semantic search
- **File Watcher component** - For live vault synchronization
- **Config module** - For server configuration
- **Document Processor module** - For initial indexing

### External Dependencies
- **FastAPI** - Web framework for REST API
- **Uvicorn** - ASGI server for running the application
- **Pydantic** - Data validation and serialization

### Integration Points
- **Startup initialization** - Performs initial vault indexing
- **Component orchestration** - Coordinates file watcher and vector store
- **MCP client communication** - Serves AI agents following MCP protocol

## Configuration Requirements

Uses shared configuration from `config/app.toml`:

```toml
[server]
host = "127.0.0.1"       # Server bind address
port = 8000              # Server port

[indexing]
chunk_size = 512         # Used during initial indexing
chunk_overlap = 64       # Used during initial indexing
quality_threshold = 0.75 # Minimum chunk quality score
```

## Usage Examples

### Starting the Server
```python
from components.mcp_server.main import main
main()  # Starts server with configured host/port
```

### Making API Requests
```python
import requests

# Query for documents
response = requests.post("http://localhost:8000/mcp/query", json={
    "query": "authentication system",
    "limit": 5
})

# Get server info
info = requests.get("http://localhost:8000/mcp/info").json()
```
