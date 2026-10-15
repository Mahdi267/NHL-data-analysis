import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from pathlib import Path

DOCS_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "raw"
)
"""Fonctions pour faire les trois tests demandés par l'énoncé """

def check_download(fetch_fn,game_id,game):
    """On a un dossier temporaire pour agir comme cache vide et forcer le téléchargment
    On a un Mock qui simule une request 'response' et dois agir comme resposne , donc retourner raise_for_status et une méthode json
    fetch_fn c'est la fonction (llm ou william) a tester.
    Adapté de la cellule 22 de demo_2/notebooks/fr/04_llm_and_rag.ipynb.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    filename = f"{game_id}.json"
    # On crée un dossier temporaire pour simuler un besoin de download
    with TemporaryDirectory() as temp_dir:
        folder =  Path(temp_dir)/"raw"
        # Avec Mock on crée une fausse réponse de l'API
        fake_response = Mock()
        fake_response.json.return_value = game
        with patch("requests.get", return_value=fake_response) as get:
            first = fetch_fn(game_id, folder)
        get.assert_called_once_with(url, timeout=30)
        fake_response.raise_for_status.assert_called_once()
        saved = json.loads((Path(folder) / filename).read_text(encoding="utf-8"))
        assert first == saved == game

def check_known_game(fetch_fn,game_id,game):
    pass
def check_cache_no_request(fetch_fn, game_id, game):
    """
    Test 2 de l'énoncé : le match est déjà en cache, donc aucune requète.
    Adapté de la cellule 22 de demo_2/notebooks/fr/04_llm_and_rag.ipynb.
    :param fetch_fn:
    :param game_id:
    :param game:
    :return:
    """
    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / "raw"
        # On fait un directory dans folder
        folder.mkdir()
        # On remplit nous-mêmes le cache avant l'appel
        (folder / f"{game_id}.json").write_text(json.dumps(game), encoding="utf-8")
        with patch("requests.get", side_effect=AssertionError("Cache was ignored : request sent")) as get:
            result = fetch_fn(game_id,folder)
        # On vérifie si Mock a été appelé. Si get a été appelé, alors la fonction a ignoré le cache
        get.assert_not_called()
        assert result == game

def check_failed_request(fetch_fn, game_id):
    pass
def check_wrong_cache(fetch_fn, game_id, game):
    pass
def run_tests(fetch_fn, game_id, game):
    pass