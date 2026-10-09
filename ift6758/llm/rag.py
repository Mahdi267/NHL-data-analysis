
from pathlib import Path
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy

def get_from_url(url):
    """Fonction qui est responsable d'aller extraire le texte depuis l'URL et de le convertir en fichier.md """

def chunks_gen(doc):
    """Fonction qui est responsable de lire le document et de transformer le texte en embeddings. Si le doc n'existe pas encore, on appel get_from_url()."""

def encode_text(text,rag):
    """Fonction qui encode les prompts de la même manière que le RAG, compare le résultat au RAG et extrait les vecteur RAG les plus similaires"""

def load_doc(url,path):
    """Retourne le teste de la doc et la met en cache sur le disque

    :param url:
    :param path:
    :return:
    """
    # Ligne tiré de demo 2
    path = Path(path)
    if not path.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    return text


def split_doc(text,chunk_size,chunk_overlap):
    """Retourne la liste des passages"""

# La recommendation de numpy vs FAISS vient de Claude, sans sources cités. Je trouve logique car semble plus rapide.
def embed(texts,embedder):
    """Retourne une matrice numpy d'embeddings normalisé"""

def retrieve(question, chunks, vector, embedder, k):
    """Retourne les k meilleurs passages avec leurs scores"""

def format_context(results):
    """Retourne la chaine en ordre pour build_messages"""