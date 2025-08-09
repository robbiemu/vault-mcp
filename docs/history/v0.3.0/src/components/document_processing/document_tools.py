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
        self, file_path: str, start_char_idx: int, end_char_idx: int
    ) -> Tuple[str, int, int]:
        """Get the full sections that enclose the given character range.

        This is the primary tool for targeted context around a RAG search.
        It identifies the tightest bounding sections and returns complete content.

        Args:
            file_path: Path to the document file
            start_char_idx: Start character position of the target range
            end_char_idx: End character position of the target range

        Returns:
            Tuple of (content, section_start_char_idx, section_end_char_idx)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            logger.debug(f"Getting enclosing sections for {file_path}")
            logger.debug(f"Character range: {start_char_idx} to {end_char_idx}")
            logger.debug(f"Document length: {len(content)} characters")

            # Find all markdown headers in the content
            headers = self._find_markdown_headers(content)

            logger.debug(f"Found {len(headers)} headers in document")
            for i, (pos, level, text) in enumerate(headers[:5]):  # Log first 5 headers
                logger.debug(
                    f"  Header {i}: pos={pos}, level={level}, text='{text[:30]}...'"
                )

            if not headers:
                # No headers found - return the relevant portion with some context
                context_start = max(0, start_char_idx - 200)
                context_end = min(len(content), end_char_idx + 200)
                logger.debug(
                    f"No headers found, returning context: {context_start} to {context_end}"
                )
                return content[context_start:context_end], context_start, context_end

            # Find the sections that enclose our character range
            enclosing_start, enclosing_end = self._find_enclosing_section_bounds(
                headers, start_char_idx, len(content)
            )

            logger.debug(f"Section bounds: {enclosing_start} to {enclosing_end}")
            logger.debug(
                f"Section length: {enclosing_end - enclosing_start} characters"
            )

            # Log a snippet of what we're returning
            snippet_start = content[enclosing_start : enclosing_start + 100].replace(
                "\n", "\\n"
            )
            snippet_end = content[
                max(enclosing_start, enclosing_end - 100) : enclosing_end
            ].replace("\n", "\\n")
            logger.debug(f"Section starts with: '{snippet_start}...'")
            logger.debug(f"Section ends with: '...{snippet_end}'")

            return (
                content[enclosing_start:enclosing_end],
                enclosing_start,
                enclosing_end,
            )

        except Exception as e:
            logger.error(f"Error getting sections from {file_path}: {e}")
            error_msg = f"Error getting sections: {e}"
            return error_msg, 0, len(error_msg)

    def _find_markdown_headers(self, content: str) -> List[Tuple[int, int, str]]:
        """Find all markdown headers in the content.

        Args:
            content: The document content

        Returns:
            List of (char_position, header_level, header_text) tuples
        """
        headers = []

        # Find ATX-style headers (# ## ### etc.)
        atx_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        for match in atx_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append((match.start(), level, text))

        # NOTE: Setext-style (underlined) headers are intentionally *not* parsed here.
        # Many vaults use plain '---' horizontal-rules inside body text, which the
        # simplistic Setext regex mis-identifies as a level-2 header and causes
        # premature section truncation (the exact issue observed with `---`).
        #
        # If your markdown relies heavily on Setext headers you can re-enable a
        # safer parser in the future, e.g. by requiring the underline to be at
        # least as long as the text line and to be followed by **another** line
        # break. For now we skip them to avoid false positives.

        # Sort headers by position
        headers.sort(key=lambda x: x[0])

        return headers

    def _find_enclosing_section_bounds(
        self,
        headers: List[Tuple[int, int, str]],
        start_char_idx: int,
        content_length: int,
    ) -> Tuple[int, int]:
        """Find the bounds of sections that enclose the given character range.

        Args:
            headers: List of header information
            start_char_idx: Start of the target character range
            content_length: Total length of the content

        Returns:
            Tuple of (section_start_char, section_end_char)
        """
        logger.debug(
            f"Finding section bounds for char starting at {start_char_idx}"
        )

        # Special case: if there are no headers, return the whole document
        if not headers:
            logger.debug("No headers found, returning whole document")
            return 0, content_length

        # Find the header that starts before or at start_char_idx
        start_header_idx = None
        for i, (header_pos, level, text) in enumerate(headers):
            if header_pos <= start_char_idx:
                start_header_idx = i
                logger.debug(
                    f"Found header before chunk start at idx {i}: pos={header_pos}, level={level}, text='{text[:30]}'"
                )
            else:
                break

        # Determine the enclosing section level
        if start_header_idx is not None:
            start_level = headers[start_header_idx][1]
            section_start = headers[start_header_idx][0]
            logger.debug(
                f"Using header {start_header_idx} as section start: pos={section_start}, level={start_level}"
            )
        else:
            logger.debug("Chunk starts before any header")
            # The chunk starts before any header
            # Find the first header that comes after the chunk
            first_header_after = None
            for i, (header_pos, level, text) in enumerate(headers):
                if header_pos > start_char_idx:
                    first_header_after = i
                    logger.debug(
                        f"First header after chunk at idx {i}: pos={header_pos}, level={level}, text='{text[:30]}'"
                    )
                    break

            if first_header_after is not None:
                # Use the level of the first header after the chunk
                start_level = headers[first_header_after][1]
                section_start = 0
                logger.debug(
                    f"Using document start as section start, level={start_level} from header {first_header_after}"
                )
            else:
                # No headers after the chunk either, use the whole document
                logger.debug("No headers found after chunk, returning whole document")
                return 0, content_length

        # Find the header that comes after end_char_idx at the same or higher level
        end_header_idx = None
        search_start_idx = start_header_idx + 1 if start_header_idx is not None else 0

        logger.debug(
            f"Looking for section end starting from header idx {search_start_idx}, target level <= {start_level}"
        )

        for i in range(search_start_idx, len(headers)):
            header_pos, level, text = headers[i]
            logger.debug(
                f"  Checking header {i}: pos={header_pos}, level={level}, text='{text[:30]}'"
            )
            # We want the next header at the same level or higher (lower number)
            # if header_pos > end_char_idx and level <= start_level:
            if header_pos > section_start and level <= start_level:
                end_header_idx = i
                logger.debug(f"Found section end at header {i}")
                break

        # Determine section end
        if end_header_idx is not None:
            section_end = headers[end_header_idx][0]
            logger.debug(f"Section ends at pos={section_end}")
        else:
            section_end = content_length
            logger.debug(
                f"No suitable end header found, section ends at document end: {section_end}"
            )

        logger.debug(
            f"Final section bounds: {section_start} to {section_end} (length: {section_end - section_start})"
        )
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
    """Tool to retrieve sections of a document based on character ranges."""

    def __init__(self) -> None:
        self.reader = DocumentReader()

    def get_enclosing_sections(
        self, file_path: str, start_char_idx: int, end_char_idx: int
    ) -> str:
        """Get the full sections that enclose the given character range.

        This is the primary tool for targeted context around a RAG search.

        Args:
            file_path: Path to the file
            start_char_idx: Start character index of the target range
            end_char_idx: End character index of the target range

        Returns:
            Content of the section(s) that enclose the character range
        """
        content, _, _ = self.reader.get_enclosing_sections(
            file_path, start_char_idx, end_char_idx
        )
        return str(content)
