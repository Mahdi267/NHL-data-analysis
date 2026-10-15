import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from pathlib import Path

DOCS_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "raw"
)
"""Fonctions pour faire les trois tests demandés par l'énoncé """

def check_download(fetch_fn,game_id,game,docs_dir=DOCS_DIR):
    """On a un dossier temporaire pour agir comme cache vide et forcer le téléchargment
    On a un Mock qui simule une request 'response' et dois agir comme resposne , donc retourner raise_for_status et une méthode json
    fetch_fn c'est la fonction (llm ou william) a tester.
    Adapté de la cellule 22 de demo_2/notebooks/fr/04_llm_and_rag.ipynb.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    filename = f"{game_id}.json"
    with TemporaryDirectory() as temp_dir:
        folder =  Path(docs_dir)
        response = Mock()
        response.json.return_value = game_id
        with patch("requests.get", return_value=response) as get:
            first = fetch_fn(game_id, folder)
        get.assert_called_once_with(
            url, timeout=30,
        )
        response.raise_for_status.assert_called_once()
        saved = json.loads((Path(folder) / filename).read_text(encoding="utf-8"))
        with patch("requests.get", side_effect=AssertionError("Cache was ignored")):
            second = fetch_fn(game_id, folder)
        assert first == second == saved == game

def check_known_game(fetch_fn,game_id,game):
    return
def check_cache_no_request(fetch_fn, game_id, game):
    return
def check_failed_request(fetch_fn, game_id):
    return
def check_wrong_cache(fetch_fn, game_id, game):
    return
def run_tests(fetch_fn, game_id, game):
    return