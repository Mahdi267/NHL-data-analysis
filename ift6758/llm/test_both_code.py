import json
from tempfile import TemporaryDirectory
from unittest import result
from unittest.mock import Mock, patch
from pathlib import Path

import requests

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def check_download(fetch_fn, game_id, game):
    """Test 1 On a un dossier temporaire pour agir comme cache vide et forcer le téléchargment
    On a un Mock qui simule une request 'response' et dois agir comme resposne , donc retourner raise_for_status et une méthode json
    fetch_fn c'est la fonction (llm ou william) a tester.
    Adapté de la cellule 22 de demo_2/notebooks/fr/04_llm_and_rag.ipynb.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    filename = f"{game_id}.json"
    # On crée un dossier temporaire pour simuler un besoin de download
    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / "raw"
        # Avec Mock on crée une fausse réponse de l'API
        fake_response = Mock()
        fake_response.json.return_value = game
        with patch("requests.get", return_value=fake_response) as get:
            first = fetch_fn(game_id, folder)
        get.assert_called_once_with(url, timeout=30)
        fake_response.raise_for_status.assert_called_once()
        saved = json.loads((Path(folder) / filename).read_text(encoding="utf-8"))
        assert first == saved == game


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
        with patch(
            "requests.get",
            side_effect=AssertionError("Cache was ignored : request sent"),
        ) as get:
            result = fetch_fn(game_id, folder)
        # On vérifie si Mock a été appelé. Si get a été appelé, alors la fonction a ignoré le cache
        get.assert_not_called()
        assert result == game


def check_failed_request(fetch_fn, game_id):
    """Test 3 (BONUS) : la requête échoue (HTTP 404), donc une erreur est levée et rien n'est mis en cache.
    Code suggéré par la cellule 23 de demo_2/notebooks/fr/04_llm_and_rag.ipynb."""
    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / "raw"
        # Avec Mock on crée une fausse réponse de l'API
        fake_response = Mock()
        # On simule une réponse d'erreur de la part de l'API
        fake_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Client Error"
        )
        with patch("requests.get", return_value=fake_response) as get:
            # On essaie la fonction
            try:
                fetch_fn(game_id, folder)
            # On pass si ca agit comme attendu
            except requests.HTTPError:  # A valider avec william
                pass
            # Sinon on avertie que problème
            else:
                raise AssertionError("No error even with HTTP flag")
        get.assert_called_once()
        assert not (folder / f"{game_id}.json").exists(), "File cached even with error"


def check_wrong_cache(fetch_fn, game_id, game):
    """Test 4 (BONUS): la match en cache est le mauvais, donc une erreur est levée.
    Cas suggéré par la cellule 23 de demo_2/notebooks/fr/04_llm_and_rag.ipynb.

    :param fetch_fn:
    :param game_id:
    :param game:
    :return:
    """
    other_id = game_id + 1
    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / "raw"
        folder.mkdir()
        # On remplit nous-mêmes le cache avec game_id mais avec le mauvais nom other_id
        (folder / f"{other_id}.json").write_text(json.dumps(game), encoding="utf-8")
        with patch(
            "requests.get",
            side_effect=AssertionError("Request sent without name error flag"),
        ) as get:
            # On essaie la fonction
            try:
                fetch_fn(other_id, folder)
            # On pass si ca agit comme attendu
            except ValueError:  # A valider avec le code de William
                pass
            # Sinon on avertie que problème
            else:
                raise AssertionError("Wrong cached match accepted without error flag")
        get.assert_not_called()


def run_tests(fetch_fn, game_id, game):
    """Run all 4 tests on fecth_fn and return results for each test. lambda idea comes from Claude

    :param fetch_fn:
    :param game_id:
    :param game:
    :return:
    """
    # On fait un dictionnaire pour pouvoir parcourir nos fonctions
    # lambda permet de créer une fonction sans non ni paramètre qui va exécuter la fonction quand elle sera appelé
    test = {
        "1. Known game": lambda: check_download(fetch_fn, game_id, game),
        "2. Cache check": lambda: check_cache_no_request(fetch_fn, game_id, game),
        "3. Failed HTTP request": lambda: check_failed_request(fetch_fn, game_id),
        "4. Wrong game in cache": lambda: check_wrong_cache(fetch_fn, game_id, game),
    }
    results = {}
    for name, test in test.items():
        try:
            test()  # Test function is run
            results[name] = "OK"
        except AssertionError as e:  # If test function failed. Before Exception because assertion is exception
            results[name] = f"ÉCHEC : {e}"
        except Exception as e:  # Recommended by Claude, allows error identification other than my Assert
            results[name] = (
                f"ERREUR : {type(e).__name__}: {e}"  # Prints the Exception type
            )
    return results
