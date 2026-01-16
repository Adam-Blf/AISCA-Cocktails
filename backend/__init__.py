"""
AISCA-Cocktails Backend Module

This module contains:
- semantic_engine: SBERT-based semantic matching (local, no API)
- rag_generator: GenAI recipe generation (Gemini/OpenAI API)
- models: Pydantic data models
"""

from .models import Cocktail, Ingredient, SearchResult
from .semantic_engine import SemanticEngine
from .rag_generator import RAGGenerator

__all__ = [
    "Cocktail",
    "Ingredient",
    "SearchResult",
    "SemanticEngine",
    "RAGGenerator",
]
