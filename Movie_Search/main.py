import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv("movies.csv")

# Load pre-trained transformer model for text embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Encode the movie plots into numerical vectors
print("Generating vector representations for plots...")
plot_vectors = embedder.encode(movies["plot"].tolist(), convert_to_tensor=False)
print("Vectorization complete.")

def movie_recommendation(user_query, results=5):
    """
    Find the most relevant movies given a user query.

    Args:
        user_query (str): Text input from the user.
        results (int): Number of top matches to retrieve.

    Returns:
        pd.DataFrame: Contains the top matching movies with similarity scores.
    """
    # Convert the query into an embedding
    query_vec = embedder.encode([user_query], convert_to_tensor=False)

    # Compute similarity between the query vector and all movie plot vectors
    similarity_scores = cosine_similarity(query_vec, plot_vectors)[0]

    # Identify indices of the top results (sorted by similarity score)
    best_idx = np.argsort(similarity_scores)[-results:][::-1]

    # Extract corresponding movies
    recommendations = movies.iloc[best_idx].copy()
    recommendations["similarity"] = [similarity_scores[i] for i in best_idx]

    return recommendations
