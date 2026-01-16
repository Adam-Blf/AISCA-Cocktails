"""
Pydantic models for AISCA-Cocktails

Data validation and serialization for cocktails, ingredients, and search results.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    """Single ingredient with quantity and optional alternatives."""

    name: str = Field(..., description="Ingredient name")
    quantity: str = Field(..., description="Quantity with unit (e.g., '50ml', '2 cuillères')")
    alternatives: list[str] = Field(default_factory=list, description="Alternative ingredients")


class Cocktail(BaseModel):
    """Complete cocktail data model."""

    id: str = Field(..., description="Unique identifier (slug)")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Short description for semantic matching")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    base_spirit: str = Field(..., description="Main spirit (rhum, vodka, gin, etc.)")
    flavor_profile: list[str] = Field(default_factory=list, description="Flavor descriptors")
    difficulty: str = Field(default="facile", description="Difficulty level")
    base_cost: float = Field(..., ge=0, description="Estimated base cost in euros")
    ingredients: list[Ingredient] = Field(default_factory=list, description="Ingredient list")
    embedding: list[float] = Field(default_factory=list, description="SBERT embedding vector (384 dims)")

    def get_text_for_embedding(self) -> str:
        """Generate text representation for SBERT encoding."""
        tags_str = ", ".join(self.tags)
        flavors_str = ", ".join(self.flavor_profile)
        return f"{self.name}. {self.description}. Tags: {tags_str}. Saveurs: {flavors_str}."


class SearchResult(BaseModel):
    """Search result with similarity score."""

    cocktail: Cocktail
    similarity_score: float = Field(..., ge=0, le=1, description="Cosine similarity score")
    rank: int = Field(..., ge=1, description="Result ranking")


class RecipeRequest(BaseModel):
    """Request for RAG recipe generation."""

    cocktail: Cocktail
    budget: float = Field(..., ge=5, le=100, description="User budget in euros")
    user_query: str = Field(..., description="Original user query/desire")


class RecipeResponse(BaseModel):
    """Generated recipe from RAG."""

    cocktail_name: str
    recipe_text: str = Field(..., description="Full generated recipe")
    adapted_ingredients: list[str] = Field(default_factory=list, description="Budget-adapted ingredients")
    tips: list[str] = Field(default_factory=list, description="Extra tips")
    estimated_cost: Optional[float] = Field(None, description="Estimated final cost")
