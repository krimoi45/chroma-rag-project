# Projet RAG avec ChromaDB

## Description
Ce projet démontre l'utilisation de ChromaDB pour la mise en place d'un système de Retrieval-Augmented Generation (RAG) en Python.

## Prérequis
- Python 3.8+
- pip

## Installation
1. Clonez le dépôt
```bash
git clone https://github.com/krimoi45/chroma-rag-project.git
cd chroma-rag-project
```

2. Créez un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
```

3. Installez les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation
Lancez le script principal :
```bash
python main.py
```

## Fonctionnalités
- Création d'une collection de documents avec ChromaDB
- Génération d'embeddings avec Sentence Transformers
- Recherche de similarité sémantique
- Exemple de système RAG basique

## Technologies
- ChromaDB
- Sentence Transformers
- NumPy
- Python

## Licence
Projet open-source
