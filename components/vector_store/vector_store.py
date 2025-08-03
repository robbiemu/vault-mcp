"""Vector store management for document embeddings and semantic search."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union, cast

import chromadb
from chromadb.config import Settings
from components.mcp_server.models import ChunkMetadata
from vault_mcp.config import EmbeddingModelConfig
from vault_mcp.embedding_factory import EmbeddingModel, create_embedding_model

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings and semantic search using ChromaDB."""

    def __init__(
        self,
        embedding_config: EmbeddingModelConfig,
        persist_directory: str = "./chroma_db",
        collection_name: str = "vault_docs",
    ):
        """Initialize the vector store.

        Args:
            embedding_config: Configuration for the embedding model
            persist_directory: Directory to persist the ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_config = embedding_config

        # Ensure the persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Initialize the embedding model using the factory
        try:
            self.embedding_model = cast(
                EmbeddingModel, create_embedding_model(embedding_config)
            )
            logger.info(
                f"Initialized embedding model: "
                f"{embedding_config.provider}/{embedding_config.model_name}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Get or create the collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except chromadb.errors.NotFoundError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Obsidian vault document chunks"},
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with text, file_path, chunk_id, and score
        """
        if not chunks:
            return

        # Extract texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        # Create metadata for each chunk
        metadatas: List[Mapping[str, Union[str, int, float, bool, None]]] = []
        for chunk in chunks:
            metadatas.append(
                {
                    "file_path": str(chunk["file_path"]),
                    "score": float(chunk["score"]),
                    "text_length": len(chunk["text"]),
                    # Add the new metadata fields
                    "start_byte": int(chunk.get("start_byte", 0)),
                    "end_byte": int(chunk.get("end_byte", 0)),
                    "original_text": str(chunk.get("original_text", "")),
                }
            )

        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)

            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=texts,
                metadatas=metadatas,
                ids=chunk_ids,
            )

            logger.info(f"Added {len(chunks)} chunks to vector store")

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def search(
        self, query: str, limit: int = 5, quality_threshold: float = 0.0
    ) -> List[ChunkMetadata]:
        """Search for relevant chunks using semantic similarity.

        Args:
            query: The search query
            limit: Maximum number of results to return
            quality_threshold: Minimum quality score for results

        Returns:
            List of ChunkMetadata objects sorted by relevance
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]

            # Search the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],  # type: ignore[arg-type]
                n_results=limit * 2,  # Get more results to filter by quality
                include=["documents", "metadatas", "distances"],
            )

            # Process results
            chunks = []
            if (
                results["documents"]
                and results["documents"][0]
                and results["metadatas"]
                and results["metadatas"][0]
                and results["distances"]
                and results["distances"][0]
            ):
                for i, (doc, metadata, _distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                        strict=False,
                    )
                ):
                    # Filter by quality threshold
                    score_value = metadata.get("score", 0)
                    if (
                        score_value is not None
                        and float(score_value) >= quality_threshold
                    ):
                        chunk_id = (
                            results["ids"][0][i]
                            if results["ids"] and results["ids"][0]
                            else f"chunk_{i}"
                        )

                        chunks.append(
                            ChunkMetadata(
                                text=str(doc),
                                file_path=str(metadata.get("file_path", "")),
                                chunk_id=str(chunk_id),
                                score=float(metadata.get("score") or 0),
                                # Populate the new fields from metadata
                                start_byte=int(metadata.get("start_byte") or 0),
                                end_byte=int(metadata.get("end_byte") or 0),
                                original_text=str(metadata.get("original_text") or ""),
                            )
                        )

            # Sort by score (quality) and limit results
            chunks.sort(key=lambda x: x.score, reverse=True)
            return chunks[:limit]

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def remove_file_chunks(self, file_path: str) -> None:
        """Remove all chunks from a specific file.

        Args:
            file_path: Path of the file whose chunks should be removed
        """
        try:
            # Get all chunks from this file
            results = self.collection.get(
                where={"file_path": file_path}, include=["metadatas"]
            )

            if results["ids"]:
                # Delete the chunks
                self.collection.delete(ids=results["ids"])
                logger.info(f"Removed {len(results['ids'])} chunks from {file_path}")

        except Exception as e:
            logger.error(f"Error removing chunks for {file_path}: {e}")

    def get_all_file_paths(self) -> List[str]:
        """Get a list of all file paths that have chunks in the vector store.

        Returns:
            List of unique file paths
        """
        try:
            # Get all documents with metadata
            results = self.collection.get(include=["metadatas"])

            if results["metadatas"]:
                file_paths = {
                    str(metadata["file_path"])
                    for metadata in results["metadatas"]
                    if isinstance(metadata.get("file_path"), str)
                }
                return sorted(list(file_paths))

            return []

        except Exception as e:
            logger.error(f"Error getting file paths: {e}")
            return []

    def get_chunk_count(self) -> int:
        """Get the total number of chunks in the vector store.

        Returns:
            Total number of chunks
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0

    def clear_all(self) -> None:
        """Clear all data from the vector store."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Obsidian vault document chunks"},
            )
            logger.info("Cleared all data from vector store")

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
