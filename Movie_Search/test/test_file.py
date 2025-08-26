import unittest
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# Try to import the function from the module, else adjust path
try:
    from movie_search import movie_recommendation
except ImportError:
    sys.path.append(os.getcwd())
    from movie_search import movie_recommendation

class MovieSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build a small in-memory dataset for test purposes
        cls.sample_data = pd.DataFrame({
            "title": ["Secret Agent", "Parisian Romance", "Action Blast"],
            "plot": [
                "An undercover agent in Paris works to stop a dangerous scheme.",
                "Two strangers fall in love under the Paris skyline.",
                "Explosive car chases and chaos in the streets of New York."
            ]
        })
        cls.model = SentenceTransformer("all-MiniLM-L6-v2")
        cls.plot_vectors = cls.model.encode(cls.sample_data["plot"].tolist(), convert_to_tensor=False)

    def test_return_type_and_columns(self):
        """Verify that the function returns a DataFrame with the right structure."""
        query = "thrilling spy story in Paris"
        output = movie_recommendation(query, results=3)
        self.assertIsInstance(output, pd.DataFrame, "Output must be a pandas DataFrame")
        expected = ["title", "plot", "similarity"]
        self.assertTrue(all(col in output.columns for col in expected),
                        f"Expected columns: {expected}")

    def test_number_of_results(self):
        """Ensure the function respects the requested number of results."""
        query = "thrilling spy story in Paris"
        k = 2
        output = movie_recommendation(query, results=k)
        self.assertEqual(len(output), k, f"Output should contain exactly {k} rows")

    def test_similarity_value_bounds(self):
        """Check if all similarity values are between 0 and 1."""
        query = "thrilling spy story in Paris"
        output = movie_recommendation(query, results=3)
        scores = output["similarity"].values
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores),
                        "All similarity values should fall within [0, 1]")

    def test_relevance_of_top_match(self):
        """Confirm that the top-ranked result is related to the query."""
        query = "thrilling spy story in Paris"
        output = movie_recommendation(query, results=1)
        top_text = output.iloc[0]["plot"].lower()
        self.assertTrue(any(word in top_text for word in ["spy", "thriller", "paris"]),
                        "Top returned movie should be relevant to the query terms")

if __name__ == "__main__":
    unittest.main()
