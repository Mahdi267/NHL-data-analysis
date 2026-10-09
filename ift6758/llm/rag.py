from pathlib import Path
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

DOCS_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "docs"
)  # Pour avoir le chemin de docs
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


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
    path = Path(Path(docs_dir) / filename)
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
    text = text.split(start, 1)[
        1
    ]  # On coupe texte à start, 1 seule coupure, on garde section 2
    # Recommendation de Claude parce que problèmes fréquents dans mes tests
    if end not in text:
        raise ValueError(f"Titre {end!r} introuvable après {start!r}")
    text = text.split(end, 1)[0]  # On coupe section 2 à end, on garde Section 2.1
    return text


def split_doc(text, chunk_size=600, chunk_overlap=120):
    """Découpe le document en passage plus court

    :param text: Document original
    :type text: str
    :param chunk_size: Taille max du morceau de texte
    :type chunk_size: int
    :param chunk_overlap: Fenètre de texte partagé entre chaque morceau de texte
    :type chunk_overlap: int
    :return: Morceaux de textes plus petit
    :rtype: list
    """
    # Possibilité de comparaison avec un split par titre MarkdownHeaderTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks
    """
    Idées Comparaison: 
    - Split par Titre
    - Chunk size différent
    - Chunk overlap différent
    """


def load_embedder(model_id=EMBED_MODEL_ID, device="cpu"):
    """Retourne le modèle sélectionné

    :param model_id: Identifiant Hugging Face, MiniLM par défaut
    :type model_id: str
    :param device: "cpu" par défaut
    :type device: str
    :return: modèle d'embedding
    :rtype: SentenceTransformer
    """
    embedder = SentenceTransformer(model_id, device=device)
    return embedder


def embed(texts, embedder):
    """Retourne une matrice numpy d'embeddings normalisé

    :param texts: Liste de textes, passages ou question a encoder
    :type texts: list[str]
    :param embedder: Modèle déjà chargé
    :type embedder: str
    :return:
    :rtype: numpy.ndarray de forme len(texts), 384 pour MiniLM
    """
    vectors = embedder.encode(texts, normalize_embeddings=True).astype("float32")
    return vectors


def retrieve(question, chunks, vectors, embedder, k=3):
    """Retourne les k meilleurs passages avec leurs scores"""
    question_embedded = embed([question], embedder)  # format (1,384)
    scores = (
        vectors @ question_embedded[0]
    )  # préfère numpy a FAISS pour comprendre comment ca marche
    positions = np.argsort(scores)[::-1][:k]
    results = [
        {
            "rank": rang,
            "index": int(index),
            "score": float(scores[index]),
            "text": chunks[index],
        }
        for rang, index in enumerate(positions, start=1)
    ]
    return results


def format_context(results):
    """Retourne la chaine en ordre pour build_messages"""
    context = "\n\n".join(
        f"[{i['rank']}] {i['text']}" for i in results
    )
    return context