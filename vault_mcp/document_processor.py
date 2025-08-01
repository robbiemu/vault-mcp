"""Document processing and chunking utilities."""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mistune


class DocumentProcessor:
    """Processes markdown documents and creates chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.markdown_parser = mistune.create_markdown(renderer="ast")

    def parse_markdown(self, content: str) -> str:
        """Parse markdown content to plain text."""
        # Parse to AST first
        ast = self.markdown_parser(content)
        # Handle case where AST might be a string or list
        if isinstance(ast, str):
            return ast
        elif isinstance(ast, list):
            return self._extract_text_from_ast(ast)
        else:
            return ""

    def _extract_text_from_ast(self, ast: List[Dict[str, Any]]) -> str:
        """Extract plain text from markdown AST."""
        text_parts = []

        for node in ast:
            if isinstance(node, dict):
                text_parts.append(self._extract_text_from_node(node))
            elif isinstance(node, str):
                text_parts.append(node)

        return " ".join(text_parts).strip()

    def _extract_text_from_node(self, node: Dict[str, Any]) -> str:
        """Extract text from a single AST node."""
        node_type = node.get("type", "")

        # Handle different node types
        if node_type in ["paragraph", "heading", "list_item", "block_quote"]:
            children = node.get("children", [])
            if children:
                return self._extract_text_from_ast(children)

        elif node_type == "text":
            raw_text = node.get("raw", "")
            return str(raw_text) if raw_text is not None else ""

        elif node_type == "link":
            # For links, we want the link text, not the URL
            children = node.get("children", [])
            if children:
                return self._extract_text_from_ast(children)

        elif node_type == "image":
            # For images, return the alt text if available
            alt_text = node.get("alt", "")
            return str(alt_text) if alt_text is not None else ""

        elif node_type in ["block_code", "inline_code"]:
            # Include code content
            raw_text = node.get("text", node.get("raw", ""))
            return str(raw_text) if raw_text is not None else ""

        elif node_type == "list":
            children = node.get("children", [])
            if children:
                return self._extract_text_from_ast(children)

        # For other node types, try to extract from children
        children = node.get("children", [])
        if children:
            return self._extract_text_from_ast(children)

        return ""

    def create_chunks(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text."""
        if not text.strip():
            return []

        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_id = f"{file_path}|{chunk_index}"
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "file_path": file_path,
                        "chunk_id": chunk_id,
                        "score": self._calculate_chunk_quality(current_chunk.strip()),
                    }
                )

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length

        # Add the final chunk if there's remaining text
        if current_chunk.strip():
            chunk_id = f"{file_path}|{chunk_index}"
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "file_path": file_path,
                    "chunk_id": chunk_id,
                    "score": self._calculate_chunk_quality(current_chunk.strip()),
                }
            )

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with more sophisticated methods
        sentence_endings = r"[.!?]+\s+"
        sentences = re.split(sentence_endings, text)

        # Clean up and filter empty sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last part of text for overlap."""
        if len(text) <= overlap_size:
            return text

        # Try to break at word boundaries
        words = text.split()
        overlap_text = ""
        for word in reversed(words):
            if len(overlap_text) + len(word) + 1 <= overlap_size:
                overlap_text = word + " " + overlap_text
            else:
                break

        return overlap_text.strip()

    def _calculate_chunk_quality(self, text: str) -> float:
        """Calculate a quality score for a chunk using heuristics."""
        if not text.strip():
            return 0.0

        score = 0.0

        # Length score (prefer medium-length chunks)
        length = len(text)
        if 100 <= length <= 800:
            score += 0.3
        elif length < 100:
            score += 0.1
        else:
            score += 0.2

        # Sentence completeness score
        sentence_endings = text.count(".") + text.count("!") + text.count("?")
        if sentence_endings > 0:
            score += 0.2

        # Structure indicators (headings, lists, etc.)
        if any(indicator in text for indicator in ["##", "###", "-", "*", "1.", "2."]):
            score += 0.2

        # Content richness (avoid chunks with mostly punctuation or very short words)
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 3:
                score += 0.2

        # Coherence (simple check for repeated topics/keywords)
        unique_words = set(word.lower() for word in words if len(word) > 3)
        if len(unique_words) / max(len(words), 1) > 0.3:  # Good word diversity
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def process_file(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a single markdown file and return its content and chunks."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # Parse markdown to plain text
            plain_text = self.parse_markdown(raw_content)

            # Create chunks
            chunks = self.create_chunks(plain_text, str(file_path))

            return raw_content, chunks

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return "", []
