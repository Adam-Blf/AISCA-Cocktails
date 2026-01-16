"""
Semantic Engine for AISCA-Cocktails

Local SBERT-based semantic search using sentence-transformers.
NO external API calls - everything runs locally.

Model: all-MiniLM-L6-v2 (384 dimensions, ~80MB)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Cocktail, SearchResult


class SemanticEngine:
    """
    Local semantic search engine using SBERT.

    Uses all-MiniLM-L6-v2 for efficient embeddings and cosine similarity
    for matching user queries to cocktails.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the semantic engine.

        Args:
            data_path: Path to cocktails.json. Defaults to data/cocktails.json.
        """
        self.model: Optional[SentenceTransformer] = None
        self.cocktails: list[Cocktail] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

        # Default data path
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "cocktails.json"
        self.data_path = data_path

    def load_model(self) -> None:
        """Load the SBERT model (downloads on first run)."""
        if self.model is None:
            print(f"Loading SBERT model: {self.MODEL_NAME}...")
            self.model = SentenceTransformer(self.MODEL_NAME)
            print("Model loaded successfully.")

    def load_data(self) -> None:
        """Load cocktails from JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.cocktails = [Cocktail(**c) for c in data.get("cocktails", [])]
        print(f"Loaded {len(self.cocktails)} cocktails from {self.data_path}")

        # Build embeddings matrix from pre-computed embeddings
        self._build_embeddings_matrix()

    def _build_embeddings_matrix(self) -> None:
        """Build numpy matrix from cocktail embeddings."""
        if not self.cocktails:
            return

        embeddings = []
        for cocktail in self.cocktails:
            if cocktail.embedding:
                embeddings.append(cocktail.embedding)
            else:
                # If no pre-computed embedding, compute it now
                self.load_model()
                text = cocktail.get_text_for_embedding()
                embedding = self.model.encode(text, convert_to_numpy=True)
                cocktail.embedding = embedding.tolist()
                embeddings.append(cocktail.embedding)

        self.embeddings_matrix = np.array(embeddings)
        print(f"Embeddings matrix shape: {self.embeddings_matrix.shape}")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a user query into an embedding vector.

        Args:
            query: User's natural language query

        Returns:
            384-dimensional embedding vector
        """
        self.load_model()
        return self.model.encode(query, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search for cocktails matching the query.

        Uses cosine similarity between query embedding and cocktail embeddings.

        Args:
            query: User's natural language query
            top_k: Number of results to return

        Returns:
            List of SearchResult with similarity scores
        """
        if self.embeddings_matrix is None:
            self.load_data()

        # Encode query
        query_embedding = self.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                SearchResult(
                    cocktail=self.cocktails[idx],
                    similarity_score=float(similarities[idx]),
                    rank=rank,
                )
            )

        return results

    def precompute_embeddings(self, save: bool = True) -> None:
        """
        Pre-compute embeddings for all cocktails and save to JSON.

        Args:
            save: Whether to save updated data back to JSON
        """
        self.load_model()
        self.load_data()

        print("Pre-computing embeddings...")
        for cocktail in self.cocktails:
            text = cocktail.get_text_for_embedding()
            embedding = self.model.encode(text, convert_to_numpy=True)
            cocktail.embedding = embedding.tolist()

        if save:
            self._save_data()
            print(f"Embeddings saved to {self.data_path}")

    def _save_data(self) -> None:
        """Save cocktails with embeddings back to JSON."""
        data = {"cocktails": [c.model_dump() for c in self.cocktails]}
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# CLI for pre-computing embeddings
if __name__ == "__main__":
    import sys

    if "--precompute" in sys.argv:
        engine = SemanticEngine()
        engine.precompute_embeddings()
    else:
        print("Usage: python -m backend.semantic_engine --precompute")
