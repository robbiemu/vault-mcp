Certainly — here is a complete, integrated technical specification for your **MCP-compliant Obsidian documentation server**, updated to include **automatic live vault syncing**.

---

# 🧾 Full Technical Specification: MCP Documentation Server (Obsidian Vault + RAG + Live Sync)

---

## 📌 Purpose

Build a **Model Context Protocol (MCP)** server that:

* Indexes a filtered subset of Markdown documents from an **Obsidian vault**
* Supports **vector-based semantic search (RAG)**
* Tracks and **scores chunks for quality**
* Allows agents to **retrieve full documents**
* Exposes a **self-describing interface** (introspection, capabilities, config)
* **Automatically re-indexes files** when they change on disk

---

## 1. 📁 Vault & Filtering

### Source

* Documents are stored in an **Obsidian vault** (i.e., a directory of `.md` files).

### Filtering

* Only include files whose **filenames begin with allowed prefixes**.
* Prefixes are defined in a config file (`config.toml`).

---

## 2. 🔧 Configuration (`config.toml`)

```toml
[paths]
vault_dir = "/absolute/path/to/obsidian-vault"

[prefix_filter]
allowed_prefixes = [
  "Resource Balance Game",
  "Decentralized Multiplayer Game System"
]

[indexing]
chunk_size = 512
chunk_overlap = 64
quality_threshold = 0.75

[watcher]
enabled = true
debounce_seconds = 2
```

---

## 3. 📦 Document Ingestion

### Steps

1. **Filter** files by prefix.
2. **Parse Markdown** into plain text (using Obsidian-friendly Markdown parser).
3. **Chunk** into overlapping segments using `chunk_size` and `chunk_overlap`.
4. **Score each chunk** using:

   * Lightweight LLM (via Ollama or llama-cpp)
   * OR heuristic scoring (e.g., based on cohesion, structure, length)
5. **Discard or reshape** low-quality chunks based on `quality_threshold`.
6. Store chunks in a **vector index** (e.g., FAISS or Chroma).

### Chunk Metadata

Each chunk stores:

```json
{
  "text": "...",
  "file_path": "Resource Balance Game - Goals.md",
  "chunk_id": "file|offset",
  "score": 0.91
}
```

---

## 4. 🔁 Vault Sync (Live Reindexing)

### Behavior

* The server **monitors the vault directory** for `.md` file changes using a file watcher.
* On file change:

  * If file is added and matches prefix → ingest & index it.
  * If file is modified → re-ingest & update it.
  * If file is deleted → remove from index.

### Implementation

* Use `watchdog` (Python library).
* Run watcher in a background thread.
* Debounce changes using `debounce_seconds` to reduce churn.

---

## 5. 🧠 Semantic Search (RAG)

* Accepts natural language queries from agents.
* Returns relevant chunks + source document info.
* Chunks are retrieved from vector index using embedding similarity.

### Query Result Format

```json
{
  "answer": "The game economy is based on...",
  "sources": [
    {
      "text": "...",
      "file_path": "Resource Balance Game - Economy.md",
      "score": 0.88
    },
    ...
  ]
}
```

---

## 6. 📄 Full Document Retrieval

* Any indexed document can be retrieved in full using its filename.
* Response includes:

  * Raw Markdown
  * Optional metadata (e.g., title, last modified)

---

## 7. 🔍 MCP Introspection & Control Interface

### Agent-accessible endpoints to expose:

* List of indexed files
* Server capabilities and configuration
* Current prefix filter
* Chunking and scoring settings

### Example `/mcp/info` Response

```json
{
  "mcp_version": "1.0",
  "capabilities": ["search", "document_retrieval", "live_sync", "introspection"],
  "indexed_files": [
    "Resource Balance Game - Goals.md",
    "Decentralized Multiplayer Game System - Consensus.md"
  ],
  "config": {
    "chunk_size": 512,
    "overlap": 64,
    "quality_threshold": 0.75
  }
}
```

---

## 8. 🌐 MCP Server Endpoints

| Method | Path            | Description                               |
| ------ | --------------- | ----------------------------------------- |
| `GET`  | `/mcp/info`     | Introspect capabilities, config, prefixes |
| `GET`  | `/mcp/files`    | List all currently indexed files          |
| `GET`  | `/mcp/document` | Return full document by `file_path` param |
| `POST` | `/mcp/query`    | Accepts query, returns RAG result         |
| `POST` | `/mcp/reindex`  | Force full re-index from disk             |

---

## 9. 🧱 Stack

| Component       | Tool                            |
| --------------- | ------------------------------- |
| Language        | Python 3.11+                    |
| Framework       | FastAPI                         |
| File watcher    | `watchdog`                      |
| Markdown parser | `mistune` or `markdown-it-py`   |
| Embeddings      | Instructor / bge-small / Ollama |
| Vector store    | Chroma (or FAISS for RAM-only)  |
| LLM interface   | LlamaIndex or LangChain         |

---

## 10. 🗂 Expected Size & Performance

* Dataset: < 24 files, \~50k tokens
* Fits in memory easily (RAM-safe)
* Cold startup: \~1–2 seconds
* Embedding & scoring is batched + cached
* File watching is low-CPU due to debouncing
