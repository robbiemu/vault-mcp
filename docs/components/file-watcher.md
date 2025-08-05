# File Watcher Component

## Purpose and Scope

The File Watcher component monitors the Obsidian vault for file system changes and automatically updates the vector store index in real-time. It provides live synchronization between the file system and the indexed document collection, ensuring search results stay current with document modifications.

## Key Interfaces and APIs

### Core Classes

- **`VaultWatcher`** - Main watcher orchestrator
  - `start()` - Begin monitoring the vault directory
  - `stop()` - Stop monitoring and cleanup resources
  - `is_running()` - Check if watcher is active

- **`VaultEventHandler`** - File system event processor
  - `on_created(event)` - Handle file creation events
  - `on_modified(event)` - Handle file modification events  
  - `on_deleted(event)` - Handle file deletion events

### Event Processing Features

- **Debouncing** - Prevents rapid-fire updates during bulk changes
- **Quality filtering** - Only indexes chunks meeting quality thresholds
- **Prefix filtering** - Respects configured filename prefix restrictions
- **Error handling** - Graceful handling of file system edge cases

## Dependencies and Integration Points

### Internal Dependencies
- **Vector Store component** - Updates document chunks in response to changes
- **Document Loader module** - Loads and processes changed files into chunks
- **Config module** - Uses watcher and filtering configuration

### External Dependencies
- **Watchdog** - File system monitoring library

### Integration Points
- **MCP Server startup** - Started automatically if enabled in configuration
- **Vault file system** - Monitors configured vault directory recursively
- **Vector index maintenance** - Keeps search index synchronized with files

## Configuration Requirements

Uses shared configuration from `config/app.toml`:

```toml
[watcher]
enabled = true           # Enable/disable file watching
debounce_seconds = 2     # Delay before processing file changes

[paths]
vault_dir = "/path/to/vault"  # Directory to monitor

[prefix_filter]
allowed_prefixes = ["Project", "Doc"]  # Only watch matching files

[indexing]
quality_threshold = 0.75  # Filter chunks by quality during updates
```

## Usage Examples

### Manual Watcher Control
```python
from components.file_watcher.file_watcher import VaultWatcher
from vault_mcp.config import load_config

config = load_config()
loader = DocumentLoader()
vector_store = VectorStore()

# Start watching
watcher = VaultWatcher(config, loader, vector_store)
watcher.start()

# Check status
if watcher.is_running():
    print("Watcher is active")

# Stop watching  
watcher.stop()
```

### Integration with Server Lifecycle
```python
# Automatic integration (from MCP server main.py)
if config.watcher.enabled:
file_watcher = VaultWatcher(config, loader, vector_store)
    file_watcher.start()
    # Watcher runs in background until server shutdown
```

### Event Processing Flow
1. **File Change Detected** → Watchdog generates event
2. **Event Filtering** → Check file extension (.md) and prefix matching
3. **Debouncing** → Wait for debounce period to handle rapid changes
4. **Processing Decision**:
   - **Created/Modified** → Process file, filter chunks, update vector store
   - **Deleted** → Remove all chunks for file from vector store
5. **Error Handling** → Log errors, continue monitoring
