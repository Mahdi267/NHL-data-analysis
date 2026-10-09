
from pathlib import Path
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy
DOCS_DIR = "ift6758/data/docs"

def load_doc(url,path):
    """Retourne le teste de la doc et la met en cache sur le disque

    :param url: Website url
    :type url: str
    :param path: /name_file.md
    :type path: str
    :return:
    """
    # Ligne tiré de demo 2
    path = Path(DOCS_DIR+path)
    if not path.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    return text
# Code adapté de démo 2
def extract_section(text,start,end):
    """Retourne le texte compris entre start et end
    
    :param text: Document complet provenant de load_doc
    :type text: str
    :param start: Titre ou la documentation commence
    :type start: str
    :param end: Titre ou la documentation termine
    :type end: str
    :return: Le texte entre les deux titres
    :rtype: str
    """
    if start not in text:
       raise ValueError(f"Titre introuvable: ",{start})
    if end not in text:
        raise ValueError(f"Titre introuvable: ",{end})
    text = text.split(start,1)[1] # On coupe texte à start, 1 seule coupure, on garde section 2
    text = text.split(end,1)[0] # On coupe section 2 à end, on garde Section 2.1
    return text

def split_doc(text,chunk_size,chunk_overlap):
    """Retourne la liste des passages"""

def embed(texts,embedder):
    """Retourne une matrice numpy d'embeddings normalisé"""

def retrieve(question, chunks, vector, embedder, k):
    """Retourne les k meilleurs passages avec leurs scores"""

def format_context(results):
    """Retourne la chaine en ordre pour build_messages"""