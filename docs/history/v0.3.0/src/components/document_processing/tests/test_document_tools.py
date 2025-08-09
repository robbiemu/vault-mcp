from components.document_processing import (
    FullDocumentRetrievalTool,
    SectionRetrievalTool,
)


def test_full_document_retrieval_tool():
    tool = FullDocumentRetrievalTool()
    content = tool.retrieve_full_document("docs/tasks/improved_heuristic.task.md")
    assert "ChunkQualityScorer" in content


def test_section_retrieval_tool():
    tool = SectionRetrievalTool()
    content = tool.get_enclosing_sections(
        "docs/tasks/improved_heuristic.task.md", 0, 150
    )
    assert "ChunkQualityScorer" in content
