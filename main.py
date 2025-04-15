import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuration du client ChromaDB
chroma_client = chromadb.Client(Settings(
    chroma_server_host="localhost",
    chroma_server_http_port=8000
))

# Création d'une collection
collection = chroma_client.create_collection(name="documents")

# Modèle d'embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Exemple de documents
documents = [
    "Le machine learning est un domaine passionnant de l'intelligence artificielle.",
    "Python est un langage de programmation très populaire pour le développement de logiciels.",
    "Les réseaux de neurones profonds permettent des avancées remarquables en traitement du langage naturel.",
    "La data science combine statistiques, programmation et analyse de données.",
    "L'apprentissage automatique trouve des applications dans de nombreux domaines comme la santé et la finance."
]

# Génération des embeddings
embeddings = model.encode(documents)

# Ajout des documents à la collection
for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
    collection.add(
        embeddings=[embedding.tolist()],
        documents=[doc],
        ids=[f"doc_{i}"]
    )

# Fonction de recherche de similarité
def recherche_similaire(requete, top_k=2):
    # Embedding de la requête
    requete_embedding = model.encode([requete])[0].tolist()
    
    # Recherche des documents similaires
    resultats = collection.query(
        query_embeddings=[requete_embedding],
        n_results=top_k
    )
    
    return resultats['documents'][0]

# Démonstration
requete_test = "Techniques avancées d'intelligence artificielle"
resultats = recherche_similaire(requete_test)

print(f"Requête : {requete_test}")
print("Documents similaires :")
for doc in resultats:
    print(f"- {doc}")

# Exemple de système RAG basique
def generer_reponse(requete):
    # Récupérer les documents similaires
    contexte = recherche_similaire(requete)
    
    # Ici, vous pourriez intégrer un modèle de génération de texte comme GPT
    reponse = f"Basé sur les documents pertinents, voici une réponse à votre requête '{requete}':\n"
    for doc in contexte:
        reponse += f"- {doc}\n"
    
    return reponse

# Exemple d'utilisation
print("\nExemple de système RAG :")
reponse_generee = generer_reponse("intelligence artificielle")
print(reponse_generee)
