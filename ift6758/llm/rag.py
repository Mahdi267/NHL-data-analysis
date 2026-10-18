"""
Simple RAG pipeline: download the documentation, split it into chunks, embed them and retrieve
the passages closest to a question.

Code partly derived from demo 2 of the IFT 6758 course and from the Hugging Face website.

Source 1: milarobotlearningcourse/data_science, demo_2/notebooks/fr/04_llm_and_rag.ipynb, commit eb3dce5 (2026-09-29)
Source 2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, all-MiniLM-L6-v2 model card (Usage)
Source 3: Claude (Anthropic), suggestions in chat, see comments
"""

from pathlib import Path
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

DOCS_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "docs"
)  # Path to the docs folder
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def load_doc(url, filename, docs_dir=DOCS_DIR):
    """Return the document text, cached on disk after the first download.

    :param url: Document URL
    :type url: str
    :param filename: Cache file name
    :type filename: str
    :param docs_dir: Cache folder
    :type docs_dir: Path or str
    :return: Document text
    :rtype: str
    :raises requests.HTTPError: If the download fails
    """
    # Adapted from source 1 (cell 9)
    path = Path(Path(docs_dir) / filename)
    if not path.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    return text


def extract_section(text, start, end):
    """Return the text between the titles start and end.

    :param text: Full document from load_doc
    :type text: str
    :param start: Title where the section starts
    :type start: str
    :param end: Title where the section ends
    :type end: str
    :return: Text between the two titles
    :rtype: str
    :raises ValueError: If a title is not found
    """
    # Adapted from source 1 (cell 9)
    # Check suggested by source 3, because of frequent problems in my tests
    if start not in text:
        raise ValueError(f"Title not found: {start!r}")
    text = text.split(start, 1)[
        1
    ]  # Cut the text at start, only once, keep the second part
    # Check suggested by source 3, because of frequent problems in my tests
    if end not in text:
        raise ValueError(f"Title {end!r} not found after {start!r}")
    text = text.split(end, 1)[0]  # Cut the second part at end, keep what comes before
    return text


def split_doc(text, chunk_size=600, chunk_overlap=120):
    """Split the document into shorter chunks.

    :param text: Original document
    :type text: str
    :param chunk_size: Max chunk size (characters)
    :type chunk_size: int
    :param chunk_overlap: Overlap between consecutive chunks
    :type chunk_overlap: int
    :return: Chunks
    :rtype: list[str]
    """
    # Adapted from source 1 (cell 9)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks
    # Possible comparison with a split by title (MarkdownHeaderTextSplitter) another time
    # Idea seen in source 2
    """
    Comparison ideas:
    - Split by title
    - Different chunk size
    - Different chunk overlap
    """


def load_embedder(model_id=EMBED_MODEL_ID, device="cpu"):
    """Load the embedding model.

    :param model_id: Hugging Face model id
    :type model_id: str
    :param device: Compute unit
    :type device: str
    :return: Embedding model
    :rtype: SentenceTransformer
    """
    # Adapted from source 1 (cell 9) and source 2
    embedder = SentenceTransformer(model_id, device=device)
    return embedder


def embed(texts, embedder):
    """Return the normalized embeddings of the texts.

    :param texts: Chunks or questions to encode
    :type texts: list[str]
    :param embedder: Loaded embedding model
    :type embedder: SentenceTransformer
    :return: Matrix of shape (len(texts), 384) for MiniLM
    :rtype: numpy.ndarray
    """
    # Adapted from source 1 (cell 9) and source 2
    vectors = embedder.encode(texts, normalize_embeddings=True).astype("float32")
    return vectors


def retrieve(question, chunks, vectors, embedder, k=3):
    """Return the k chunks closest to the question (cosine similarity).

    :param question: Question to search for
    :type question: str
    :param chunks: Chunks from split_doc
    :type chunks: list[str]
    :param vectors: Chunk embeddings from embed
    :type vectors: numpy.ndarray
    :param embedder: Loaded embedding model
    :type embedder: SentenceTransformer
    :param k: Number of chunks to return
    :type k: int
    :return: rank, index, score and text of each chunk
    :rtype: list[dict]
    """
    # Adapted from source 1 (cell 11), with numpy instead of FAISS
    question_embedded = embed([question], embedder)  # shape (1, 384)
    scores = (
        vectors @ question_embedded[0]
    )  # numpy rather than FAISS, to understand how it works
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


def format_context(results, source=None):
    """Join the retrieved chunks into a context string for build_messages.

    :param results: Output of retrieve
    :type results: list[dict]
    :param source: Source name added to each chunk
    :type source: str or None
    :return: Context
    :rtype: str
    """
    # Adapted from source 1 (cell 11)
    # context = "\n\n".join(
    #    f"[{i['rank']}]{i['text']}" for i in results
    # )
    parts = []
    for r in results:
        if source:
            parts.append(
                f"[{r['rank']}] (source:{source}, passage {r['index']}) {r['text']}"
            )
        else:
            parts.append(f"[{r['rank']}] (index {r['index']}) {r['text']}")
    context = "\n\n".join(parts)
    return context
