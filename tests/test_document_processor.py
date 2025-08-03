"""Tests for document processing and chunking."""

from vault_mcp.document_processor import DocumentProcessor


def test_markdown_parsing():
    """Test markdown parsing to plain text."""
    processor = DocumentProcessor()

    markdown_content = """# Test Document

This is a paragraph with **bold** and *italic* text.

## Section

- List item 1
- List item 2

```python
def hello():
    return "world"
```

[Link text](https://example.com)
"""

    plain_text = processor.parse_markdown(markdown_content)

    # Verify key content is extracted
    assert "Test Document" in plain_text
    assert "paragraph" in plain_text
    assert "bold" in plain_text
    assert "italic" in plain_text
    assert "Section" in plain_text
    assert "List item 1" in plain_text
    assert "def hello" in plain_text
    assert "Link text" in plain_text


def test_chunk_creation():
    """Test creating chunks from text."""
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    text = (
        "This is sentence one. This is sentence two. This is sentence three. "
        "This is sentence four. This is sentence five."
    )

    chunks = processor.create_chunks(text, "test.md")

    assert len(chunks) > 1  # Should create multiple chunks
    assert all(chunk["file_path"] == "test.md" for chunk in chunks)
    # Roughly chunk_size + some buffer
    assert all(len(chunk["text"]) <= 120 for chunk in chunks)
    assert all("score" in chunk for chunk in chunks)
    assert all("chunk_id" in chunk for chunk in chunks)


def test_chunk_quality_scoring():
    """Test chunk quality scoring heuristics."""
    processor = DocumentProcessor()

    # High quality chunk - medium length, complete sentences, structure
    high_quality = (
        "This is a comprehensive overview of the system. It includes multiple "
        "complete sentences with good structure. The content is informative and "
        "well-organized."
    )
    score_high = processor._calculate_chunk_quality(high_quality)

    # Low quality chunk - very short
    low_quality = "Short."
    score_low = processor._calculate_chunk_quality(low_quality)

    # Empty chunk
    empty_chunk = ""
    score_empty = processor._calculate_chunk_quality(empty_chunk)

    assert score_high > score_low
    assert score_low > score_empty
    assert score_empty == 0.0


def test_sentence_splitting():
    """Test sentence splitting functionality."""
    processor = DocumentProcessor()

    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    sentences = processor._split_into_sentences(text)

    assert len(sentences) == 4
    assert "First sentence" in sentences[0]
    assert "Second sentence" in sentences[1]
    assert "Third sentence" in sentences[2]
    assert "Fourth sentence" in sentences[3]


def test_overlap_text_extraction():
    """Test overlap text extraction for chunks."""
    processor = DocumentProcessor()

    text = "This is a long sentence with many words that should be used for overlap."
    overlap = processor._get_overlap_text(text, 20)

    assert len(overlap) <= 20
    assert overlap.endswith("for overlap.")  # Should end with last complete word


def test_process_file(sample_markdown_files):
    """Test processing a complete markdown file."""
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

    raw_content, chunks = processor.process_file(sample_markdown_files["matching"])

    assert raw_content.startswith("# Resource Balance Game Overview")
    assert len(chunks) > 0
    assert all(
        chunk["file_path"] == str(sample_markdown_files["matching"]) for chunk in chunks
    )
    assert all(chunk["score"] > 0 for chunk in chunks)

    # Verify all chunks have the new metadata fields
    for chunk in chunks:
        assert "start_byte" in chunk
        assert "end_byte" in chunk
        assert "original_text" in chunk
        assert isinstance(chunk["start_byte"], int)
        assert isinstance(chunk["end_byte"], int)
        assert isinstance(chunk["original_text"], str)
        assert 0 <= chunk["start_byte"] < chunk["end_byte"]
        assert chunk["end_byte"] <= len(raw_content)


def test_process_empty_file(sample_markdown_files):
    """Test processing an empty markdown file."""
    processor = DocumentProcessor()

    raw_content, chunks = processor.process_file(sample_markdown_files["empty"])

    assert raw_content == ""
    assert len(chunks) == 0


def test_process_nonexistent_file(tmp_path):
    """Test processing a file that doesn't exist."""
    processor = DocumentProcessor()
    nonexistent_file = tmp_path / "nonexistent.md"

    raw_content, chunks = processor.process_file(nonexistent_file)

    assert raw_content == ""
    assert len(chunks) == 0


def test_raw_chunking_with_byte_positions():
    """Test the new raw chunking method that preserves byte positions."""
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

    raw_text = "# Title\n\nThis is **bold** text. More content here."
    file_path = "test.md"

    chunks = processor._create_chunks_from_raw(raw_text, file_path)

    # Verify we have chunks
    assert len(chunks) > 0

    # Verify each chunk has the new fields
    for chunk in chunks:
        assert "start_byte" in chunk
        assert "end_byte" in chunk
        assert "original_text" in chunk
        assert "text" in chunk  # Clean parsed text
        assert "file_path" in chunk
        assert "chunk_id" in chunk
        assert "score" in chunk

        # Verify byte positions make sense
        assert 0 <= chunk["start_byte"] < len(raw_text)
        assert chunk["start_byte"] < chunk["end_byte"]
        assert chunk["end_byte"] <= len(raw_text)

        # Verify original text matches the byte positions
        expected_original = raw_text[chunk["start_byte"] : chunk["end_byte"]]
        assert chunk["original_text"] == expected_original

        # Verify clean text is different from original (markdown should be parsed)
        if "**" in chunk["original_text"]:
            assert "**" not in chunk["text"]  # Bold markdown should be processed
            assert "bold" in chunk["text"]  # But content should remain


def test_chunks_preserve_byte_continuity():
    """Test that chunks with overlap still maintain correct byte positions."""
    processor = DocumentProcessor(chunk_size=20, chunk_overlap=5)

    raw_text = "This is a long text that should be split into multiple chunks."
    chunks = processor._create_chunks_from_raw(raw_text, "test.md")

    # Should have multiple chunks due to small chunk size
    assert len(chunks) > 1

    # Verify chunks are in order and have correct byte positions
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        # Current chunk should end before or at the next chunk's start + overlap
        assert current_chunk["end_byte"] > next_chunk["start_byte"]

        # But start positions should progress forward
        assert current_chunk["start_byte"] < next_chunk["start_byte"]
