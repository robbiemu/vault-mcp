"""Document exploration tools for retrieving full documents and sections."""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class DocumentReader:
    """Encapsulates document reading functionality with section parsing."""

    def __init__(self) -> None:
        """Initialize the document reader."""
        pass

    def read_full_document(self, file_path: str) -> str:
        """Read and return the entire content of a document.

        WARNING: This can return a large amount of text and should be used as a
        last resort when more targeted methods are insufficient.

        Args:
            file_path: Path to the document file

        Returns:
            The complete document content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return f"Error reading document: {e}"

    def get_enclosing_sections(
        self, file_path: str, start_byte: int, end_byte: int
    ) -> str:
        """Get the full sections that enclose the given byte range.

        This is the primary tool for targeted context around a RAG search.
        It identifies the tightest bounding sections and returns complete content.

        Args:
            file_path: Path to the document file
            start_byte: Start byte position of the target range
            end_byte: End byte position of the target range

        Returns:
            Content of the section(s) that enclose the byte range
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Find all markdown headers in the content
            headers = self._find_markdown_headers(content)

            if not headers:
                # No headers found - return the relevant portion with some context
                context_start = max(0, start_byte - 200)
                context_end = min(len(content), end_byte + 200)
                return content[context_start:context_end]

            # Find the sections that enclose our byte range
            enclosing_start, enclosing_end = self._find_enclosing_section_bounds(
                headers, start_byte, end_byte, len(content)
            )

            return content[enclosing_start:enclosing_end]

        except Exception as e:
            logger.error(f"Error getting sections from {file_path}: {e}")
            return f"Error getting sections: {e}"

    def _find_markdown_headers(self, content: str) -> List[Tuple[int, int, str]]:
        """Find all markdown headers in the content.

        Args:
            content: The document content

        Returns:
            List of (byte_position, header_level, header_text) tuples
        """
        headers = []

        # Find ATX-style headers (# ## ### etc.)
        atx_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        for match in atx_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append((match.start(), level, text))

        # Find Setext-style headers (underlined with = or -)
        setext_pattern = re.compile(r"^(.+)\n([=-]{3,})$", re.MULTILINE)
        for match in setext_pattern.finditer(content):
            underline_char = match.group(2)[0]
            level = 1 if underline_char == "=" else 2
            text = match.group(1).strip()
            headers.append((match.start(), level, text))

        # Sort headers by position
        headers.sort(key=lambda x: x[0])

        return headers

    def _find_enclosing_section_bounds(
        self,
        headers: List[Tuple[int, int, str]],
        start_byte: int,
        end_byte: int,
        content_length: int,
    ) -> Tuple[int, int]:
        """Find the bounds of sections that enclose the given byte range.

        Args:
            headers: List of header information
            start_byte: Start of the target byte range
            end_byte: End of the target byte range
            content_length: Total length of the content

        Returns:
            Tuple of (section_start_byte, section_end_byte)
        """
        # Find the header that starts before or at start_byte
        start_header_idx = None
        for i, (header_pos, _level, _text) in enumerate(headers):
            if header_pos <= start_byte:
                start_header_idx = i
            else:
                break

        # Find the header that comes after end_byte at the same or higher level
        end_header_idx = None
        if start_header_idx is not None:
            start_level = headers[start_header_idx][1]
            for i in range(start_header_idx + 1, len(headers)):
                header_pos, level, text = headers[i]
                if header_pos > end_byte and level <= start_level:
                    end_header_idx = i
                    break

        # Determine section bounds
        if start_header_idx is not None:
            section_start = headers[start_header_idx][0]
        else:
            section_start = 0

        if end_header_idx is not None:
            section_end = headers[end_header_idx][0]
        else:
            section_end = content_length

        return section_start, section_end


class FullDocumentRetrievalTool:
    """Tool to retrieve full document content."""

    def __init__(self) -> None:
        self.reader = DocumentReader()

    def retrieve_full_document(self, file_path: str) -> str:
        """Retrieve entire document content.

        WARNING: This can return a large amount of text and should be used as a
        last resort when more targeted methods are insufficient.

        Args:
            file_path: Path to the file

        Returns:
            Entire content of the document
        """
        return str(self.reader.read_full_document(file_path))


class SectionRetrievalTool:
    """Tool to retrieve sections of a document based on byte ranges."""

    def __init__(self) -> None:
        self.reader = DocumentReader()

    def get_enclosing_sections(
        self, file_path: str, start_byte: int, end_byte: int
    ) -> str:
        """Get the full sections that enclose the given byte range.

        This is the primary tool for targeted context around a RAG search.

        Args:
            file_path: Path to the file
            start_byte: Start byte of the target range
            end_byte: End byte of the target range

        Returns:
            Content of the section(s) that enclose the byte range
        """
        return str(self.reader.get_enclosing_sections(file_path, start_byte, end_byte))
