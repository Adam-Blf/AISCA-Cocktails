# AISCA-Cocktails

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![SBERT](https://img.shields.io/badge/Sentence--Transformers-FFD21E?logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?logo=google&logoColor=white)
![EFREI](https://img.shields.io/badge/EFREI-Paris-000091)

## Description

**AISCA-Cocktails** (AI Semantic Cocktail Advisor) est une application de recommandation de cocktails basée sur l'analyse sémantique des envies de l'utilisateur. Le système utilise un modèle SBERT local pour le matching sémantique et une GenAI (RAG) pour générer des recettes personnalisées adaptées au budget.

> Projet étudiant EFREI - Architecture IA avec moteur sémantique local et RAG.

---

## Architecture Technique

```
AISCA-Cocktails/
├── README.md                 # Documentation du projet
├── requirements.txt          # Dépendances Python
├── .env.example              # Template variables d'environnement
├── .gitignore                # Fichiers à ignorer
│
├── data/
│   └── cocktails.json        # Référentiel de cocktails (embeddings pré-calculés)
│
├── backend/
│   ├── __init__.py           # Module Python
│   ├── semantic_engine.py    # Moteur SBERT + Similarité Cosinus
│   ├── rag_generator.py      # Générateur RAG (API Gemini/OpenAI)
│   └── models.py             # Modèles de données Pydantic
│
└── frontend/
    └── app.py                # Application Streamlit
```

---

## Flux de Données

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AISCA-Cocktails                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐         ┌────────────────────────────────────────────┐   │
│   │   FRONTEND   │         │                 BACKEND                    │   │
│   │  (Streamlit) │         │                                            │   │
│   │              │         │   ┌────────────────────────────────────┐   │   │
│   │  ┌────────┐  │  envie  │   │     SEMANTIC ENGINE (Local)        │   │   │
│   │  │ Input  │──┼─────────┼──►│  ┌──────────────────────────────┐  │   │   │
│   │  │ User   │  │         │   │  │  SBERT (all-MiniLM-L6-v2)    │  │   │   │
│   │  └────────┘  │         │   │  │  - Encode query              │  │   │   │
│   │              │         │   │  │  - Cosine Similarity         │  │   │   │
│   │  ┌────────┐  │         │   │  │  - Top-K matching            │  │   │   │
│   │  │ Budget │  │         │   │  └──────────────────────────────┘  │   │   │
│   │  │ Slider │  │         │   │              │                      │   │   │
│   │  └────────┘  │         │   │              ▼                      │   │   │
│   │              │         │   │  ┌──────────────────────────────┐   │   │   │
│   │              │         │   │  │      cocktails.json          │   │   │   │
│   │              │         │   │  │  (référentiel + embeddings)  │   │   │   │
│   │              │         │   │  └──────────────────────────────┘   │   │   │
│   │              │         │   └────────────────────────────────────┘   │   │
│   │              │         │                   │                         │   │
│   │              │         │   cocktail trouvé │                         │   │
│   │              │         │                   ▼                         │   │
│   │              │         │   ┌────────────────────────────────────┐   │   │
│   │              │         │   │      RAG GENERATOR (API)           │   │   │
│   │  ┌────────┐  │  recette│   │  ┌──────────────────────────────┐  │   │   │
│   │  │ Recette│◄─┼─────────┼───│  │  Google Gemini / OpenAI      │  │   │   │
│   │  │ Card   │  │         │   │  │  - Contexte: cocktail data   │  │   │   │
│   │  └────────┘  │         │   │  │  - Prompt: recette + budget  │  │   │   │
│   │              │         │   │  │  - Output: recette détaillée │  │   │   │
│   └──────────────┘         │   │  └──────────────────────────────┘  │   │   │
│                            │   └────────────────────────────────────┘   │   │
│                            └────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stack Technique

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| **Frontend** | Streamlit | Interface rapide, adaptée aux prototypes data/ML |
| **Semantic Engine** | sentence-transformers (SBERT) | Modèle local, pas de dépendance API pour le matching |
| **Modèle SBERT** | `all-MiniLM-L6-v2` | Léger (80MB), performant, multilingue |
| **Similarité** | Cosine Similarity | Standard pour les embeddings textuels |
| **Data Store** | JSON local | Simple, portable, pas de setup DB |
| **RAG GenAI** | Google Gemini / OpenAI | Génération de texte de qualité |
| **Validation** | Pydantic | Typage fort et validation des données |

---

## Contraintes Techniques (Cahier des charges EFREI)

### 1. Moteur Sémantique 100% Local
- **Aucun appel API** pour le matching sémantique
- Utilisation exclusive de `sentence-transformers`
- Modèle : `all-MiniLM-L6-v2` (pré-entraîné)
- Calcul de similarité cosinus avec `sklearn` ou `numpy`

### 2. Données Locales
- Référentiel de cocktails en **JSON**
- Embeddings pré-calculés stockés dans le JSON
- Pas de base de données (PostgreSQL, MongoDB, etc.)

### 3. GenAI uniquement pour RAG
- Appel API **après** le matching sémantique
- Utilisé pour :
  - Générer la recette détaillée
  - Adapter au budget de l'utilisateur
  - Proposer des alternatives

---

## Installation

### Prérequis
- Python 3.10+
- pip ou uv

### Setup

```bash
# Cloner le projet
git clone https://github.com/Adam-Blf/AISCA-Cocktails.git
cd AISCA-Cocktails

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec votre clé API (Gemini ou OpenAI)
```

### Pré-calcul des embeddings (optionnel)

```bash
python -m backend.semantic_engine --precompute
```

---

## Lancement

```bash
# Lancer l'application Streamlit
streamlit run frontend/app.py
```

L'application sera accessible sur `http://localhost:8501`

---

## Utilisation

1. **Décrivez votre envie** : "Je veux quelque chose de fruité et rafraîchissant pour l'été"
2. **Définissez votre budget** : Utilisez le slider (5€ - 50€)
3. **Obtenez votre recommandation** :
   - Le système trouve les cocktails correspondants (SBERT)
   - La GenAI génère une recette adaptée à votre budget

---

## Structure des Données

### cocktails.json

```json
{
  "cocktails": [
    {
      "id": "mojito",
      "name": "Mojito",
      "description": "Cocktail cubain rafraîchissant à base de rhum, menthe et citron vert",
      "tags": ["rafraîchissant", "été", "menthe", "rhum", "citron"],
      "base_spirit": "rhum",
      "flavor_profile": ["frais", "acidulé", "sucré"],
      "difficulty": "facile",
      "base_cost": 8,
      "ingredients": [
        {"name": "Rhum blanc", "quantity": "50ml", "alternatives": ["Rhum ambré"]},
        {"name": "Menthe fraîche", "quantity": "10 feuilles"},
        {"name": "Citron vert", "quantity": "1"},
        {"name": "Sucre de canne", "quantity": "2 cuillères"},
        {"name": "Eau gazeuse", "quantity": "100ml"}
      ],
      "embedding": [0.023, -0.156, ...]  // Vecteur 384 dimensions
    }
  ]
}
```

---

## API GenAI (RAG)

### Prompt Template

```
Tu es un expert en mixologie. Voici un cocktail recommandé :

**Cocktail**: {cocktail_name}
**Description**: {description}
**Ingrédients de base**: {ingredients}

L'utilisateur a un budget de {budget}€.

Génère une recette détaillée en adaptant les ingrédients si nécessaire :
1. Liste des ingrédients avec quantités précises
2. Instructions étape par étape
3. Conseils de présentation
4. Alternatives économiques si budget serré

Sois créatif mais réaliste avec le budget indiqué.
```

---

## Variables d'Environnement

```env
# .env
GENAI_PROVIDER=gemini  # ou "openai"
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

---

## Features

- [x] Architecture projet définie
- [ ] Référentiel de cocktails (50+ cocktails)
- [ ] Moteur sémantique SBERT
- [ ] Interface Streamlit
- [ ] Intégration RAG Gemini
- [ ] Adaptation budget
- [ ] Tests unitaires

---

## Roadmap

### Phase 1 - MVP
- Référentiel de 20 cocktails
- Matching sémantique basique
- Génération de recette simple

### Phase 2 - Enrichissement
- 50+ cocktails
- Filtres avancés (alcool, difficulté)
- Historique des recommandations

### Phase 3 - Polish
- UI/UX améliorée
- Mode sombre
- Export PDF de la recette

---

## Auteur

**Adam Beloucif** - Projet EFREI 2024-2025

---

## Licence

Ce projet est réalisé dans un cadre académique (EFREI).


---

<p align="center">
  <sub>Par <a href="https://adam.beloucif.com">Adam Beloucif</a> · Data Engineer & Fullstack Developer · <a href="https://github.com/Adam-Blf">GitHub</a> · <a href="https://www.linkedin.com/in/adambeloucif/">LinkedIn</a></sub>
</p>
