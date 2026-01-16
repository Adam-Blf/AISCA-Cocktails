"""
RAG Generator for AISCA-Cocktails

Generates personalized cocktail recipes using GenAI (Gemini or OpenAI).
Uses cocktail data as context (RAG) to produce budget-adapted recipes.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from .models import Cocktail, RecipeRequest, RecipeResponse

# Load environment variables
load_dotenv()


class RAGGenerator:
    """
    Recipe generator using RAG (Retrieval-Augmented Generation).

    Supports Google Gemini and OpenAI as backends.
    """

    PROMPT_TEMPLATE = """Tu es un expert en mixologie et barman professionnel.

Voici le cocktail recommandé basé sur les envies de l'utilisateur :

**Cocktail** : {cocktail_name}
**Description** : {description}
**Spiritueux de base** : {base_spirit}
**Profil de saveurs** : {flavor_profile}
**Difficulté** : {difficulty}
**Coût estimé de base** : {base_cost}€

**Ingrédients originaux** :
{ingredients_list}

---

**Demande de l'utilisateur** : "{user_query}"
**Budget disponible** : {budget}€

---

Génère une recette détaillée et personnalisée en suivant ce format :

## Recette : {cocktail_name}

### Ingrédients (adaptés au budget)
- Liste les ingrédients avec quantités précises
- Si le budget est serré, propose des alternatives économiques
- Si le budget est généreux, suggère des versions premium

### Préparation
1. Étapes numérotées et claires
2. Inclure les techniques de mixologie
3. Temps de préparation estimé

### Présentation
- Verre recommandé
- Garniture
- Conseils de service

### Astuces du barman
- 2-3 conseils pour réussir ce cocktail
- Variantes possibles

### Estimation du coût
- Coût approximatif avec les ingrédients suggérés

Sois créatif, précis et adapte-toi au budget indiqué !"""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the RAG generator.

        Args:
            provider: "gemini" or "openai". Defaults to env GENAI_PROVIDER.
        """
        self.provider = provider or os.getenv("GENAI_PROVIDER", "gemini")
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate GenAI client."""
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel("gemini-1.5-flash")
            print("Gemini client initialized")

        except ImportError:
            raise ImportError("google-generativeai package not installed")

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            self.client = OpenAI(api_key=api_key)
            print("OpenAI client initialized")

        except ImportError:
            raise ImportError("openai package not installed")

    def _format_ingredients(self, cocktail: Cocktail) -> str:
        """Format ingredients list for the prompt."""
        lines = []
        for ing in cocktail.ingredients:
            line = f"- {ing.name}: {ing.quantity}"
            if ing.alternatives:
                alts = ", ".join(ing.alternatives)
                line += f" (alternatives: {alts})"
            lines.append(line)
        return "\n".join(lines)

    def _build_prompt(self, request: RecipeRequest) -> str:
        """Build the prompt from request data."""
        cocktail = request.cocktail
        return self.PROMPT_TEMPLATE.format(
            cocktail_name=cocktail.name,
            description=cocktail.description,
            base_spirit=cocktail.base_spirit,
            flavor_profile=", ".join(cocktail.flavor_profile),
            difficulty=cocktail.difficulty,
            base_cost=cocktail.base_cost,
            ingredients_list=self._format_ingredients(cocktail),
            user_query=request.user_query,
            budget=request.budget,
        )

    def generate_recipe(self, request: RecipeRequest) -> RecipeResponse:
        """
        Generate a personalized recipe using RAG.

        Args:
            request: RecipeRequest with cocktail, budget, and user query

        Returns:
            RecipeResponse with generated recipe
        """
        prompt = self._build_prompt(request)

        if self.provider == "gemini":
            recipe_text = self._generate_gemini(prompt)
        else:
            recipe_text = self._generate_openai(prompt)

        return RecipeResponse(
            cocktail_name=request.cocktail.name,
            recipe_text=recipe_text,
            adapted_ingredients=[],  # Could be parsed from response
            tips=[],  # Could be parsed from response
            estimated_cost=None,  # Could be parsed from response
        )

    def _generate_gemini(self, prompt: str) -> str:
        """Generate text using Gemini."""
        response = self.client.generate_content(prompt)
        return response.text

    def _generate_openai(self, prompt: str) -> str:
        """Generate text using OpenAI."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en mixologie qui génère des recettes de cocktails détaillées.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        return response.choices[0].message.content


# Quick test
if __name__ == "__main__":
    from .models import Ingredient

    # Test cocktail
    test_cocktail = Cocktail(
        id="mojito",
        name="Mojito",
        description="Cocktail cubain rafraîchissant",
        tags=["rafraîchissant", "été"],
        base_spirit="rhum",
        flavor_profile=["frais", "acidulé"],
        difficulty="facile",
        base_cost=8,
        ingredients=[
            Ingredient(name="Rhum blanc", quantity="50ml"),
            Ingredient(name="Menthe", quantity="10 feuilles"),
        ],
    )

    request = RecipeRequest(
        cocktail=test_cocktail,
        budget=15,
        user_query="Je veux quelque chose de frais pour l'été",
    )

    generator = RAGGenerator()
    response = generator.generate_recipe(request)
    print(response.recipe_text)
