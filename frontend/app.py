"""
AISCA-Cocktails Frontend

Streamlit application for cocktail recommendations using semantic search
and RAG-based recipe generation.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.semantic_engine import SemanticEngine
from backend.rag_generator import RAGGenerator
from backend.models import RecipeRequest

# Page configuration
st.set_page_config(
    page_title="AISCA-Cocktails",
    page_icon="🍹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .cocktail-card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .recipe-box {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_semantic_engine():
    """Load and cache the semantic engine."""
    engine = SemanticEngine()
    engine.load_data()
    return engine


@st.cache_resource
def load_rag_generator():
    """Load and cache the RAG generator."""
    try:
        return RAGGenerator()
    except Exception as e:
        st.warning(f"RAG Generator non disponible: {e}")
        return None


def render_header():
    """Render the app header."""
    st.markdown('<h1 class="main-header">AISCA-Cocktails</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Recommandation intelligente de cocktails basée sur vos envies</p>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with filters and settings."""
    with st.sidebar:
        st.markdown("## Paramètres")

        budget = st.slider(
            "Budget (€)",
            min_value=5,
            max_value=50,
            value=15,
            step=1,
            help="Votre budget pour les ingrédients",
        )

        top_k = st.slider(
            "Nombre de résultats",
            min_value=1,
            max_value=10,
            value=3,
            help="Nombre de cocktails à afficher",
        )

        st.markdown("---")
        st.markdown("### À propos")
        st.markdown(
            """
            **AISCA-Cocktails** utilise :
            - 🧠 **SBERT** pour la recherche sémantique
            - 🤖 **GenAI** pour les recettes personnalisées

            *Projet EFREI 2024-2025*
            """
        )

        return budget, top_k


def render_cocktail_card(result, index, expanded=False):
    """Render a single cocktail result card."""
    cocktail = result.cocktail
    score = result.similarity_score

    with st.expander(f"#{index} {cocktail.name} - Score: {score:.1%}", expanded=expanded):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Description:** {cocktail.description}")
            st.markdown(f"**Base:** {cocktail.base_spirit.capitalize()}")
            st.markdown(f"**Difficulté:** {cocktail.difficulty.capitalize()}")
            st.markdown(f"**Coût estimé:** {cocktail.base_cost}€")

        with col2:
            st.markdown("**Tags:**")
            tags_html = " ".join([f"`{tag}`" for tag in cocktail.tags[:5]])
            st.markdown(tags_html)

            st.markdown("**Saveurs:**")
            flavors_html = " ".join([f"`{f}`" for f in cocktail.flavor_profile])
            st.markdown(flavors_html)

        st.markdown("**Ingrédients:**")
        for ing in cocktail.ingredients:
            alt_text = f" (alt: {', '.join(ing.alternatives)})" if ing.alternatives else ""
            st.markdown(f"- {ing.name}: {ing.quantity}{alt_text}")

    return cocktail


def render_recipe(recipe_response):
    """Render the generated recipe."""
    st.markdown("---")
    st.markdown("## Recette Générée")
    st.markdown(
        f'<div class="recipe-box">{recipe_response.recipe_text}</div>',
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""
    render_header()
    budget, top_k = render_sidebar()

    # Load engines
    with st.spinner("Chargement du moteur sémantique..."):
        engine = load_semantic_engine()

    rag_generator = load_rag_generator()

    # Main input
    st.markdown("### Décrivez vos envies")
    user_query = st.text_area(
        "Que souhaitez-vous boire ?",
        placeholder="Ex: Je veux quelque chose de fruité et rafraîchissant pour une soirée d'été...",
        height=100,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        search_clicked = st.button("🔍 Trouver mon cocktail", use_container_width=True)

    # Search and display results
    if search_clicked and user_query:
        with st.spinner("Analyse sémantique en cours..."):
            results = engine.search(user_query, top_k=top_k)

        if results:
            st.markdown("---")
            st.markdown(f"### Cocktails recommandés ({len(results)} résultats)")

            # Store results in session state
            st.session_state["search_results"] = results
            st.session_state["user_query"] = user_query
            st.session_state["budget"] = budget

            for i, result in enumerate(results, 1):
                render_cocktail_card(result, i, expanded=(i == 1))

            # Generate recipe button
            st.markdown("---")
            if rag_generator:
                st.markdown("### Générer une recette détaillée")
                selected_idx = st.selectbox(
                    "Choisir un cocktail",
                    range(len(results)),
                    format_func=lambda x: results[x].cocktail.name,
                )

                if st.button("📝 Générer la recette adaptée à mon budget"):
                    selected_cocktail = results[selected_idx].cocktail
                    request = RecipeRequest(
                        cocktail=selected_cocktail,
                        budget=budget,
                        user_query=user_query,
                    )

                    with st.spinner("Génération de la recette avec l'IA..."):
                        try:
                            recipe = rag_generator.generate_recipe(request)
                            render_recipe(recipe)
                        except Exception as e:
                            st.error(f"Erreur lors de la génération: {e}")
            else:
                st.info(
                    "Configurez votre clé API dans `.env` pour activer la génération de recettes."
                )
        else:
            st.warning("Aucun cocktail trouvé. Essayez une autre description.")

    elif search_clicked and not user_query:
        st.warning("Veuillez décrire vos envies pour trouver un cocktail.")

    # Quick suggestions
    st.markdown("---")
    st.markdown("### Suggestions rapides")
    suggestions = [
        "Quelque chose de rafraîchissant pour l'été",
        "Un cocktail fort et classique",
        "Une boisson fruitée et colorée",
        "Un apéritif léger et pétillant",
    ]

    cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(f"💡 {suggestion[:20]}...", key=f"sugg_{i}", use_container_width=True):
                st.session_state["suggestion"] = suggestion
                st.rerun()


if __name__ == "__main__":
    main()
