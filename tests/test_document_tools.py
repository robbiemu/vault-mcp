from vault_mcp.document_tools import FullDocumentRetrievalTool, SectionRetrievalTool


def test_full_document_retrieval_tool():
    tool = FullDocumentRetrievalTool()
    content = tool.retrieve_full_document("docs/tasks/rewrite_agent_tooling.task.md")
    assert "User Story:" in content


def test_section_retrieval_tool():
    tool = SectionRetrievalTool()
    content = tool.get_enclosing_sections(
        "docs/tasks/rewrite_agent_tooling.task.md", 0, 150
    )
    assert "User Story:" in content
