import unittest

from components.embedding_system import create_embedding_model
from vault_mcp.config import load_config


class TestEmbeddingSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_embedding_model_creation(self):
        # Test creating the embedding model
        model = create_embedding_model(self.config.embedding_model)
        self.assertIsNotNone(model)
        print(f"Model created: {type(model).__name__}")

    def test_instruction_tuned_query_formatting(self):
        # Create the embedding model
        model = create_embedding_model(self.config.embedding_model)

        # Assuming the model supports _get_query_embedding (which it does)
        custom_query = '{"instruction": "Retrieve facts", "query": "AI history"}'
        result_query = model._get_query_embedding(custom_query)
        print(f"Processed query: {result_query}")


if __name__ == "__main__":
    unittest.main()
