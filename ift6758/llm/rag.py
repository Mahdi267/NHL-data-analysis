from pathlib import Path
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy

DOCS_DIR = Path(__file__).resolve().parents[1]/"data"/"docs" # Pour avoir le chemin de docs


def load_doc(url, filename, docs_dir=DOCS_DIR):
    """Retourne le texte de la doc et la met en cache sur le disque

    :param url: Website url
    :type url: str
    :param filename: nom du fichier
    :type filename: str
    :param docs_dir: (optionnel) chemin vers le dossier
    :type docs_dir: Path ou str
    :return: texte du document
    :rtype: str
    :raises requests.HTTPError: Téléchargement échoue
    """
    # Ligne tiré de demo 2
    path = Path(Path(docs_dir)/filename)
    if not path.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    return text


# Code adapté de démo 2
def extract_section(text, start, end):
    """Retourne le texte compris entre start et end

    :param text: Document complet provenant de load_doc
    :type text: str
    :param start: Titre où la section commence
    :type start: str
    :param end: Titre où la section termine
    :type end: str
    :raises ValueError: Retourne une erreur avec le titre introuvable
    :return: Le texte entre les deux titres
    :rtype: str
    """
    # Recommendation de Claude parce que problèmes fréquents dans mes tests
    if start not in text:
        raise ValueError(f"Titre introuvable: {start!r}")
    text = text.split(start, 1)[1]

    # On coupe texte à start, 1 seule coupure, on garde section 2
    # Recommendation de Claude parce que problèmes fréquents dans mes tests
    if end not in text:
        raise ValueError(f"Titre {end!r} introuvable après {start!r}")
    text = text.split(end, 1)[0]  # On coupe section 2 à end, on garde Section 2.1
    return text


def split_doc(text, chunk_size, chunk_overlap):
    """Retourne la liste des passages"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)
    return chunks


def embed(texts, embedder):
    """Retourne une matrice numpy d'embeddings normalisé"""


def retrieve(question, chunks, vector, embedder, k):
    """Retourne les k meilleurs passages avec leurs scores"""


def format_context(results):
    """Retourne la chaine en ordre pour build_messages"""
