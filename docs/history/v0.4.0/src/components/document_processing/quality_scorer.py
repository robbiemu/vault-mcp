"""Content-based quality scoring for text chunks.

This module provides a revised heuristic for evaluating chunk quality that focuses
entirely on content evaluation rather than structural validation, since our new
MarkdownNodeParser pipeline ensures structurally sound chunks.
"""

import logging

logger = logging.getLogger(__name__)


class ChunkQualityScorer:
    """
    Calculates a quality score for a text chunk using content-based heuristics.
    Assumes the chunk is already structurally coherent.
    """

    def score(self, text: str) -> float:
        """Calculate a quality score for a chunk using revised heuristics.

        The scoring is based on three content-focused factors:
        1. Optimal Length (0.4 points): Prefers information-rich but not
        excessive length
        2. Content Richness (0.3 points): Rewards substantial vocabulary
        3. Information Density (0.3 points): Rewards diversity of meaningful words

        Args:
            text: The text chunk to score

        Returns:
            A quality score between 0.0 and 1.0
        """
        if not text.strip():
            return 0.0

        words = text.split()
        num_words = len(words)
        if num_words < 3:  # Filter out extremely short/useless chunks
            return 0.0

        # --- REVISED FACTOR 1: Optimal Length (0.4 points) ---
        # Prefers chunks that are information-rich but not excessively long.
        length_score = 0.0
        num_chars = len(text)
        if 150 <= num_chars <= 1024:  # Ideal range for a dense paragraph
            length_score = 0.4
        elif 50 <= num_chars < 150:  # A decent sentence
            length_score = 0.2
        else:  # Acceptable, but not ideal (too short or too long)
            length_score = 0.1

        # --- REVISED FACTOR 2: Content Richness (0.3 points) ---
        # Rewards chunks with more substantial vocabulary.
        richness_score = 0.0
        avg_word_length = sum(len(word) for word in words) / num_words
        if avg_word_length > 4.5:  # Increased threshold for "rich"
            richness_score = 0.3
        elif avg_word_length > 3.5:
            richness_score = 0.15

        # --- REVISED FACTOR 3: Information Density (0.3 points) ---
        # Rewards chunks with a good diversity of meaningful words.
        density_score = 0.0
        unique_words = set(word.lower() for word in words if len(word) > 3)
        diversity_ratio = len(unique_words) / num_words
        if diversity_ratio > 0.6:  # Increased threshold for "dense"
            density_score = 0.3
        elif diversity_ratio > 0.4:
            density_score = 0.15

        # The final score is the sum of the content-based factors.
        final_score = length_score + richness_score + density_score
        return min(final_score, 1.0)
