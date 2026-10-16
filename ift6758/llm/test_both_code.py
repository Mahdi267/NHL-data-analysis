import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from pathlib import Path

import requests

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# Idée suggérée par Claude. D'ou la nomenclature _fonction
def _http_error(status):
    """Build an HTTPError like raise_for_status() does, with response.status_code set."""
    return requests.HTTPError(f"{status} Error", response=Mock(status_code=status))


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
            side_effect=AssertionError(
                "File in cache was ignored : request for an existing file (in cache) was sent"
            ),
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
        assert not (folder / f"{game_id}.json").exists(), (
            "File cached even if the client returned an error"
        )


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
                raise AssertionError(
                    f"cache file {other_id}.json contained a game whose id is {game_id}; the function returned it instead of raising ValueError"
                )
        get.assert_not_called()

# Le raisonnement derrière les fonctions de tests de saison à été élaboré avec Claude pour aider à valider la structure
# des fausses données.

def check_season_download(fetch_season_fn):
    """Fonction qui simule un faux fetch game qui renvoie 3 match suivi d'un 404. On vérifie si la fonction
    fetch_season_llm s'arrêt bien après et produit un bon output. (3 dict,IDs int, en ordre)

    :return:
    """
    season = 2016
    existing = [2016020001, 2016020002, 2016020003]  # Match in fake season
    calls = []  # list for requested game_id

    # Fake fetch_game because we want to test fetch_season and not receive errors from real fetch_game
    def fake_fetch_game(game_id, cache_dir):
        calls.append(game_id)
        if game_id in existing:
            return {"id": game_id, "plays": []}
        raise _http_error(404)

    with TemporaryDirectory() as temp_dir:
        # Fonction by Claude to replace real fetch_game by fake on in fetch_season
        with patch.dict(
            fetch_season_fn.__globals__, {"fetch_game_llm": fake_fetch_game}
        ):
            result = fetch_season_fn(season, 2, Path(temp_dir))

    # Check if 4 calls and no more beacuase of 404
    assert calls == existing + [2016020004], (
        f"Regular season must stop at the first 404 (game 2016020004). Calls made: {calls}"
    )
    # Check if game id are int, like in first prompt
    assert all(type(i) is int for i in calls), (
        f"Each game_id must be an int, got: {calls}"
    )
    ids = [g["id"] for g in result]  # Get returned match id
    assert ids == existing, f"Expected the 3 games {existing} sorted by id, got: {ids}"


def check_playoff_structure(fetch_season_fn):
    """Check if 8, 4, 2, 1 rule an pass next serie at 404 is fufilled. Compare answer with known
    series match list and check for wrong calls ex(0191,0251, etc)
    :return:
    """
    matchups = {1: 8, 2: 4, 3: 2, 4: 1}
    # Match in fake serie with proper matchup structure
    existing = [
        2016030111,
        2016030121,
        2016030131,
        2016030141,
        2016030151,
        2016030161,
        2016030171,
        2016030181,
        2016030211,
        2016030221,
        2016030231,
        2016030241,
        2016030311,
        2016030321,
        2016030411,
    ]
    # Extra match that should not be called
    forbidden = [2016030191, 2016030251, 2016030331, 2016030421]
    calls = []  # list for requested game_id

    # Fake fetch_game because we want to test fetch_season and not receive errors from real fetch_game
    def fake_fetch_game(game_id, cache_dir):
        calls.append(game_id)
        if game_id in existing:
            return {"id": game_id, "plays": []}
        raise _http_error(404)

    with TemporaryDirectory() as temp_dir:
        # Fonction by Claude to replace real fetch_game by fake on in fetch_season
        with patch.dict(
            fetch_season_fn.__globals__, {"fetch_game_llm": fake_fetch_game}
        ):
            result = fetch_season_fn(2016, 3, Path(temp_dir))

    # unexisting serie must not be called
    wrong = [i for i in calls if i in forbidden]
    assert not wrong, (
        f"Rounds 1, 2, 3, 4 have 8, 4, 2, 1 matchups, but these non-existing series were requested: {wrong}"
    )
    # 15 match, function visits each series
    ids = [g["id"] for g in result]
    assert ids == existing, (
        f"Expected one game per series ({len(existing)} games), got {len(ids)}: {ids}"
    )


def check_error_prop(fetch_season_fn):
    """Same as chec_failed_request, sceks if error propagates

    :return:
    """
    calls = []  # list for requested game_id

    # Fake fetch_game because we want to test fetch_season and not receive errors from real fetch_game
    def fake_fetch_game(game_id, cache_dir):
        calls.append(game_id)
        if game_id == 2016020001:
            return {"id": game_id, "plays": []}
        raise _http_error(500)

    with TemporaryDirectory() as temp_dir:
        # Fonction by Claude to replace real fetch_game by fake on in fetch_season
        with patch.dict(
            fetch_season_fn.__globals__, {"fetch_game_llm": fake_fetch_game}
        ):
            try:  # Ty function
                fetch_season_fn(2016, 2, Path(temp_dir))
            except requests.HTTPError as e:  # Check if 500 received
                assert e.response.status_code == 500, (
                    f"Expected the HTTP 500 error to propagate, got status {e.response.status_code}"
                )
            else:  # Else warn qwen that error did not propagate
                raise AssertionError(
                    "An HTTP 500 error was caught and treated like a missing game (404). Only 404 means the game does not exist; any other error must propagate."
                )
    # Function must stop at error
    assert calls == [2016020001, 2016020002], (
        f"The function must stop when an error propagates. Calls made: {calls}"
    )


def check_unexisting_season(fetch_season_fn):
    """Checks handeling of none existing season. Suppose to receive []
    :return:
    """
    calls = []  # list for requested game_id

    # Fake fetch_game because we want to test fetch_season and not receive errors from real fetch_game
    def fake_fetch_game(game_id, cache_dir):
        calls.append(game_id)
        raise _http_error(404)

    with TemporaryDirectory() as temp_dir:
        # Fonction by Claude to replace real fetch_game by fake on in fetch_season
        with patch.dict(
            fetch_season_fn.__globals__, {"fetch_game_llm": fake_fetch_game}
        ):
            regular = fetch_season_fn(2030, 2, Path(temp_dir))
            regular_calls = list(calls)
            playoffs = fetch_season_fn(2030, 3, Path(temp_dir))
    # Regular season = one call then stop at 404
    assert regular_calls == [2030020001], (
        f"Regular season must stop at the first 404. Calls made: {regular_calls}"
    )
    assert regular == [], (
        f"A season with no games must return an empty list, got: {regular}"
    )
    # Serie = no match = playoff empty
    assert playoffs == [], (
        f"Playoffs with no games must return an empty list, got: {playoffs}"
    )


def check_game_in_cache(fetch_season_fn):
    """Test to see if season with games already in cache request for said games or not. Should use game in cache and
    not request API.

    :return:
    """
    cached = [2016020001, 2016020002]
    url = "https://api-web.nhle.com/v1/gamecenter/2016020003/play-by-play"

    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / "raw"
        folder.mkdir()
        # we fill cache with 2 valid match
        for game_id in cached:
            (folder / f"{game_id}.json").write_text(
                json.dumps({"id": game_id, "plays": []}), encoding="utf-8"
            )
        # Fake server response with mock
        fake_response = Mock()
        fake_response.raise_for_status.side_effect = _http_error(404)
        with patch("requests.get", return_value=fake_response) as get:
            result = fetch_season_fn(2016, 2, folder)

        get.assert_called_once_with(url, timeout=30)
        assert not (folder / "2016020003.json").exists(), (
            "File cached for a game that returned 404"
        )
        ids = [g["id"] for g in result]
        assert ids == cached, f"Expected the 2 cached games {cached}, got: {ids}"

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

def run_season_tests(fetch_season_fn):
    """Run the 5 season tests on fetch_season_fn and return results for each test.
    Same structure as run_tests.
    """
    tests = {
        "1. Regular season stops at first 404": lambda: check_season_download(fetch_season_fn),
        "2. Playoff structure 8-4-2-1": lambda: check_playoff_structure(fetch_season_fn),
        "3. Non-404 error propagates": lambda: check_error_prop(fetch_season_fn),
        "4. Non-existing season returns []": lambda: check_unexisting_season(fetch_season_fn),
        "5. Integration with cache": lambda: check_game_in_cache(fetch_season_fn),
    }
    results = {}
    for name, test in tests.items():
        try:
            test()
            results[name] = "OK"
        except AssertionError as e:
            results[name] = f"ÉCHEC : {e}"
        except Exception as e:
            results[name] = f"ERREUR : {type(e).__name__}: {e}"
    return results