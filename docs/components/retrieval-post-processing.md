# Retrieval Post-processing

This document describes the retrieval post-processing mechanisms within the `vault-mcp` server, which enhance the relevance and quality of retrieved document chunks. It covers both the `agentic` (AI-enhanced) and `static` (deterministic) retrieval modes, leveraging LlamaIndex for efficient context handling.

## Purpose

The retrieval post-processing system is designed to:

- **Enhance Retrieved Context**: Improve the quality and relevance of document chunks retrieved from the vector store.
- **Support Multiple Modes**: Provide two distinct post-processing modes:
    - **`agentic` Mode**: Utilizes AI agents and LLMs to rewrite and refine chunks for richer, more comprehensive answers.
    - **`static` Mode**: Expands chunks to their full section context, providing fast and deterministic retrieval without LLM interaction.
- **Integrate Seamlessly**: Work within the overall query processing pipeline to deliver contextually relevant information to the MCP client.

## Architecture

### Key Classes & Concepts

- **`ChunkRewriterPostprocessor`**: This post-processor is used in `agentic` mode. It orchestrates the **sequential** rewriting of each retrieved chunk, providing the other chunks as context for each operation.

- **`StaticContextPostprocessor`**: This post-processor is used in `static` mode. It expands retrieved chunks to include their full contextual section (e.g., the entire paragraph or the section defined by a heading), providing a direct and un-rewritten context.

- **`create_agentic_query_engine`**: A factory function that configures and returns a LlamaIndex query engine. It dynamically selects and integrates the appropriate post-processor (`ChunkRewriterPostprocessor` or `StaticContextPostprocessor`) based on the `retrieval.mode` setting in the application's configuration. It also handles the setup of embedding models, LLM configurations, and vector store interactions.

## Retrieval Workflows

The system supports two primary retrieval workflows, determined by the `retrieval.mode` configuration:

### 1. Agentic Mode Workflow

This mode is designed for high-quality, AI-enhanced responses, suitable for complex queries requiring nuanced understanding and synthesis.

1.  **Initial Retrieval**: The query engine retrieves relevant document chunks from the vector store based on semantic similarity.
2.  **Chunk Rewriting**: The `ChunkRewriterPostprocessor` takes these retrieved chunks and, for each chunk, initiates an AI agent (e.g., `ChunkRewriteAgent`). This agent interacts with an LLM to rewrite or expand the chunk, making it more directly relevant to the query and potentially incorporating additional context.
3.  **Direct Presentation**: The rewritten chunks are returned directly as the source nodes in the response. The system is configured **not** to perform a final synthesis step that would combine them into a single answer.

### 2. Static Mode Workflow

This mode prioritizes speed and determinism, providing direct contextual information without LLM-based rewriting. It's ideal for scenarios where raw, expanded context is sufficient.

1.  **Initial Retrieval**: Similar to agentic mode, relevant document chunks are retrieved from the vector store.
2.  **Context Expansion**: The `StaticContextPostprocessor` expands each retrieved chunk to include its full surrounding section (e.g., the entire paragraph, list item, or section defined by a heading). This ensures that the chunk is presented within its original, complete context.
3.  **Direct Presentation**: The expanded, un-rewritten chunks are then returned as the relevant context for the query. No further LLM interaction for content refinement occurs at this stage.

## Usage

The `Agentic Retriever` is pivotal to server operations, providing the primary retrieval and refinement capabilities leveraged by the REST API endpoints. It integrates seamlessly within the FastAPI lifecycle, ensuring efficient search and generation functionality.

## Conclusion

The `Agentic Retriever` component embodies the core intelligence of the vault-mcp server, offering sophisticated RAG features. Its design enables seamless integration with other system components, maintaining code modularity, readability, and extensibility.

---

