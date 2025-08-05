"""Tests for chunk quality scoring functionality."""

from components.document_processing import ChunkQualityScorer


class TestChunkQualityScorer:
    """Test cases for ChunkQualityScorer."""

    def test_score_empty_text(self):
        """Test scoring empty text returns 0."""
        scorer = ChunkQualityScorer()
        assert scorer.score("") == 0.0
        assert scorer.score("   ") == 0.0

    def test_score_very_short_text(self):
        """Test scoring very short text returns 0."""
        scorer = ChunkQualityScorer()
        assert scorer.score("hi") == 0.0
        assert scorer.score("a b") == 0.0

    def test_score_optimal_length_text(self):
        """Test scoring text in optimal length range."""
        scorer = ChunkQualityScorer()

        # Create text in optimal range (150-1024 chars)
        optimal_text = (
            "This is a well-structured paragraph that contains substantial information "
            "about document processing systems and their implementation details. "
        ) * 2

        score = scorer.score(optimal_text)
        assert score > 0.3  # Should get at least optimal length score

    def test_score_content_richness(self):
        """Test scoring based on vocabulary richness."""
        scorer = ChunkQualityScorer()

        # Rich vocabulary text
        rich_text = (
            "The comprehensive documentation system implements sophisticated "
            "algorithms for processing heterogeneous data structures efficiently."
        )

        # Poor vocabulary text
        poor_text = "The the the and and and but but but yes yes yes no no no ok ok ok."

        rich_score = scorer.score(rich_text)
        poor_score = scorer.score(poor_text)

        assert rich_score > poor_score

    def test_score_information_density(self):
        """Test scoring based on information density."""
        scorer = ChunkQualityScorer()

        # High diversity text
        diverse_text = (
            "Advanced machine learning algorithms utilize sophisticated neural network "
            "architectures implementing backpropagation optimization techniques."
        )

        # Low diversity text (repeated words)
        repetitive_text = (
            "test test test test test test test test test test test test test "
            "test test test"
        )

        diverse_score = scorer.score(diverse_text)
        repetitive_score = scorer.score(repetitive_text)

        assert diverse_score > repetitive_score

    def test_score_combined_factors(self):
        """Test scoring with all factors contributing positively."""
        scorer = ChunkQualityScorer()

        # High-quality text: optimal length, rich vocabulary, diverse content
        high_quality = """
        Advanced document processing systems implement sophisticated natural language
        processing techniques to analyze textual content efficiently. These systems
        utilize machine learning algorithms for semantic understanding and information
        extraction from heterogeneous data sources.
        """

        score = scorer.score(high_quality)
        assert score >= 0.6  # Should score well on multiple factors

    def test_score_boundaries(self):
        """Test scoring at different length boundaries."""
        scorer = ChunkQualityScorer()

        # Test different length categories
        short_text = (
            "Short text with good vocabulary sophistication but insufficient length."
        )
        medium_text = (
            "This medium-length text contains sophisticated vocabulary and "
            "demonstrates reasonable information density with diverse linguistic "
            "patterns and technical terminology throughout."
        )
        long_text = (
            "This is an extremely long text that exceeds the optimal range "
            "for document processing. "
        ) * 20

        short_score = scorer.score(short_text)
        medium_score = scorer.score(medium_text)
        long_score = scorer.score(long_text)

        # Medium should typically score better than extremes
        assert medium_score >= short_score
        assert medium_score >= long_score

    def test_score_max_value(self):
        """Test that score never exceeds 1.0."""
        scorer = ChunkQualityScorer()

        # Create theoretically perfect text
        perfect_text = """
        Advanced computational linguistics algorithms utilize sophisticated neural
        network architectures implementing backpropagation optimization techniques
        for semantic understanding. These methodologies demonstrate exceptional
        performance characteristics across heterogeneous linguistic datasets.
        """

        score = scorer.score(perfect_text)
        assert score <= 1.0
