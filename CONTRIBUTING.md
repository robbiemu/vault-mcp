# Contributing to Vault MCP

Thank you for your interest in contributing to Vault MCP! We welcome contributions from everyone, whether you're fixing bugs, adding features, improving documentation, or helping with testing.

## Getting Started

### Prerequisites

This project uses **[uv](https://docs.astral.sh/uv/)** for fast, reliable Python package management. Install uv first:

```bash
# Install uv (recommended method)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv

# Or via Homebrew (macOS)
brew install uv
```

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub first, then:
   git clone https://github.com/YOUR_USERNAME/vault-mcp.git
   cd vault-mcp
   ```

2. **Set Up Development Environment**
   ```bash
   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate  # Unix/macOS
   # or
   .venv\Scripts\activate     # Windows

   # Install with development dependencies
   uv sync --extra dev
   ```

3. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   pytest

   # Start the server to test functionality
   vault-mcp
   ```

## How to Contribute

### 1. Choose Your Contribution Type

**🐛 Bug Fixes**
- Look for issues labeled `bug` or `good first issue`
- Check existing issues before creating new ones
- Include reproduction steps and environment details

**✨ New Features**
- Discuss large features in an issue first
- Follow the existing architecture patterns
- Update documentation and tests

**📚 Documentation**
- Improve existing docs
- Add examples and use cases
- Fix typos and unclear explanations

**🧪 Testing**
- Add test coverage for untested code
- Create integration tests
- Performance and stress testing

### 2. Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Your Changes**
   - Follow the project structure and coding standards
   - Write or update tests for your changes
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run the full test suite
   pytest

   # Run specific tests
   pytest tests/test_specific_module.py

   # Test with different configurations
   pytest -x --tb=short
   ```

4. **Code Quality Checks**
   ```bash
   # Format code
   black components/ shared/ vault_mcp/
   ruff check --fix components/ shared/ vault_mcp/

   # Type checking
   mypy components/ shared/ vault_mcp/

   # Security scanning
   bandit -c pyproject.toml -r components/ shared/ vault_mcp/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add semantic search improvements"
   # or
   git commit -m "fix: resolve file watcher memory leak"
   ```

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Project Architecture

Understanding the project structure will help you contribute effectively:

```
vault-mcp/
├── components/              # Discrete system components
│   ├── api_app/             # Standard REST API server
│   ├── mcp_app/             # MCP-compliant server wrapper
│   ├── vault_service/       # Core business logic service
│   ├── vector_store/        # Document embedding and search
│   ├── file_watcher/        # Live file monitoring
│   ├── document_processing/ # Document loading and processing
│   ├── embedding_system/    # Pluggable embedding models
│   └── agentic_retriever/   # Agentic retrieval and chunk rewriting
├── shared/                  # Shared utilities and libraries
│   ├── config.py            # Configuration management
│   ├── initializer.py       # Service initialization
│   └── tests/               # Shared component tests
├── vault_mcp/               # Main application entry point
│   └── main.py              # CLI and server orchestration
├── tests/                   # Root-level integration tests
├── config/                  # Configuration files
│   ├── app.toml             # Main application config
│   └── prompts.toml         # AI/LLM prompts
└── docs/                    # Documentation
```

### Key Design Principles

1. **Component-Based Architecture**: Each component is self-contained with its own tests
2. **Unified Service Layer**: `VaultService` handles all business logic
3. **Pluggable Systems**: Embedding models and document sources are extensible
4. **Configuration-Driven**: Behavior controlled via TOML configuration
5. **Test Coverage**: All components should have comprehensive tests

## Code Standards

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use `black` for formatting (line length: 88 characters)
- Use `ruff` for linting
- Use type hints (`mypy` for type checking)

### Commit Message Format
Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(embedding): add MLX embedding support for Apple Silicon"
git commit -m "fix(watcher): resolve memory leak in file monitoring"
git commit -m "docs(config): add examples for Joplin integration"
```

### Testing Guidelines

1. **Write Tests for New Code**
   - Unit tests for individual functions/classes
   - Integration tests for component interactions
   - End-to-end tests for user workflows

2. **Test Structure**
   ```python
   # tests/test_component.py
   import pytest
   from components.your_component import YourComponent

   class TestYourComponent:
       def test_basic_functionality(self):
           # Arrange
           component = YourComponent()
           
           # Act
           result = component.do_something()
           
           # Assert
           assert result == expected_value

       def test_error_handling(self):
           # Test error conditions
           pass
   ```

3. **Test Configuration**
   - Use `pytest.ini` or `pyproject.toml` for test configuration
   - Mock external dependencies (APIs, file systems)
   - Use fixtures for common test data

### Documentation Standards

1. **Code Documentation**
   ```python
   def search_documents(query: str, limit: int = 10) -> List[SearchResult]:
       """Search indexed documents using semantic similarity.
       
       Args:
           query: The search query string
           limit: Maximum number of results to return
           
       Returns:
           List of SearchResult objects sorted by relevance
           
       Raises:
           ValueError: If query is empty or limit is invalid
       """
   ```

2. **README Updates**
   - Keep feature lists current
   - Update installation instructions
   - Add new configuration examples

3. **Configuration Documentation**
   - Document new configuration options
   - Provide examples for different use cases
   - Include troubleshooting information

## Testing Your Contribution

### Local Testing

1. **Unit Tests**
   ```bash
   # Run all tests
   pytest

   # Run tests with coverage
   pytest --cov=components --cov=shared --cov=vault_mcp

   # Run specific test files
   pytest tests/test_vault_service.py
   ```

2. **Integration Testing**
   ```bash
   # Test with different document sources
   pytest -k "test_obsidian or test_joplin"

   # Test with different embedding models
   pytest -k "test_embedding"
   ```

3. **Manual Testing**
   ```bash
   # Start the server
   vault-mcp

   # Test API endpoints
   curl http://localhost:8000/files
   curl -X POST http://localhost:8000/query \
        -H "Content-Type: application/json" \
        -d '{"query": "test search"}'
   ```

### Performance Testing

```bash
# Profile memory usage
python -m memory_profiler vault_mcp/main.py

# Load testing (if applicable)
ab -n 100 -c 10 http://localhost:8000/files
```

## Submitting Your Contribution

### Pull Request Guidelines

1. **PR Title**: Use conventional commit format
2. **Description**: Include:
   - What changes you made
   - Why you made them
   - How to test the changes
   - Any breaking changes

3. **Checklist**:
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Type hints added
   - [ ] Breaking changes documented

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Type hints added
```

## Reporting Issues

### Bug Reports

Include the following information:

```markdown
## Environment
- OS: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- Python version: [e.g., 3.11.5]
- Vault MCP version: [e.g., 0.4.0]
- Document source: [Standard/Obsidian/Joplin]

## Describe the Bug
Clear and concise description of the bug.

## Steps to Reproduce
1. Configure the server with...
2. Start the server...
3. Make a request to...
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Logs
```
Include relevant log output
```

## Configuration
```toml
# Include relevant parts of your config/app.toml
```
```

### Feature Requests

```markdown
## Is your feature request related to a problem?
Clear description of the problem.

## Describe the solution you'd like
Clear description of what you want to happen.

## Describe alternatives you've considered
Alternative solutions or features you've considered.

## Additional context
Screenshots, mockups, or additional information.
```

## Development Resources

### Useful Commands

```bash
# Development server with auto-reload
vault-mcp --serve-api  # API only for testing

# Run with debug logging
export LOG_LEVEL=DEBUG && vault-mcp

# Profile startup time
time vault-mcp --serve-api

# Check dependency security
uv pip audit
```

### Debugging Tips

1. **Use Debug Logging**
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.debug("Debugging information here")
   ```

2. **Component Testing**
   ```bash
   # Test individual components
   python -m components.vault_service.tests.test_main
   ```

3. **Configuration Validation**
   ```python
   from shared.config import load_config
   config = load_config("config/app.toml")
   print(config)  # Check parsed configuration
   ```

### Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers)

## Community

### Getting Help

1. **GitHub Issues**: For bugs and feature requests
2. **GitHub Discussions**: For questions and community interaction
3. **Code Review**: Pull request discussions

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive
- Help others learn and grow
- Focus on what is best for the community
- Show empathy towards other community members

## Recognition

Contributors will be recognized in:
- GitHub contributor graphs
- Release notes for significant contributions
- README acknowledgments section

## License

By contributing to Vault MCP, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the `question` label
- Start a discussion in GitHub Discussions
- Review existing issues and discussions for similar questions

Thank you for helping make Vault MCP better! 🚀