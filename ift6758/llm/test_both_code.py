"""
Tests used to compare the Qwen-generated functions (fetch_game_llm, fetch_season_llm) with William's manual code.
Each run_* function returns {test name: "OK" | "FAIL : msg" | "ERROR : Type: msg"}.

Source 1: milarobotlearningcourse/data_science, demo_2/notebooks/fr/04_llm_and_rag.ipynb, commit eb3dce5 (2026-09-29)
Source 2: William Perktold, NHLyear class, ift6758/notebooks/raw_data_acquisition.ipynb
Source 3: Claude (Anthropic), test code written and reviewed with Claude in chat, see comments
"""
import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from pathlib import Path

import requests

# DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DOCS_DIR = Path("../data")

# This translation layer was made by CLAUDE
def make_william_fetch(nhl_year_class):
    """Wrap NHLyear.download (source 2) into the fetch_game_llm signature, raising HTTPError on failure.

    :param nhl_year_class: William's NHLyear class
    :type nhl_year_class: type
    :return: fetch(game_id, cache_dir) -> game
    :rtype: callable
    """
    def william_fetch_game(game_id, cache_dir):
        # 4 digit int or string corresponding to the year the season starts (ex: 2023 for the 2023-2024 season)
        season = nhl_year_class(str(game_id)[:4], cache_dir)
        status_code, game = season.download(game_id, cache_dir)
        # William returns the http status code AND the json content (play by play): we translate
        if str(status_code) != "200":
            raise http_error(int(status_code))
        return game

    return william_fetch_game

# This translation layer was made by CLAUDE
def william_cache_file(folder, game_id):
    """Return William's cache path: cache_dir/<year>/<game_id>.json.

    :param folder: Cache folder
    :type folder: Path
    :param game_id: Game id
    :type game_id: int
    :return: Cache file path
    :rtype: Path
    """
    return folder / str(game_id)[:4] / f"{game_id}.json"

def http_error(status):
    """Build an HTTPError like raise_for_status() does.

    :param status: HTTP status code
    :type status: int
    :return: Error with response.status_code set
    :rtype: requests.HTTPError
    """
    return requests.HTTPError(f"{status} Error", response=Mock(status_code=status))

# Source 3 proposed to add cache_file(folder, gid)
def check_download(fetch_fn, game_id, game, cache_file=lambda folder, gid: folder / f"{gid}.json"):
    """Test 1: empty cache, the game is downloaded (mocked), saved and returned.

    :param fetch_fn: Function to test (LLM or William)
    :type fetch_fn: callable
    :param game_id: Game id
    :type game_id: int
    :param game: Reference game
    :type game: dict
    :param cache_file: Cache path builder, different for William
    :type cache_file: callable
    :return: None
    """
    # Adapted from source 1 (cell 22)
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    filename = f"{game_id}.json"
    # Temporary folder to force a download
    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / str(game_id)[:4]
        # Fake API response with Mock
        fake_response = Mock()
        fake_response.json.return_value = game
        fake_response.status_code = 200  # William : read status_code instead of raise_for_status
        with patch("requests.get", return_value=fake_response) as get:
            first = fetch_fn(game_id, folder)
        get.assert_called_once_with(url, timeout=30)
        fake_response.raise_for_status.assert_called_once()
        saved = json.loads(cache_file(folder, game_id).read_text(encoding="utf-8"))  # William
        assert first == saved == game



def check_cache_no_request(fetch_fn, game_id, game, cache_file=lambda folder, gid: folder / f"{gid}.json"):
    """Test 2: the game is already in cache, so no request is sent.

    :param fetch_fn: Function to test
    :type fetch_fn: callable
    :param game_id: Game id
    :type game_id: int
    :param game: Reference game
    :type game: dict
    :param cache_file: Cache path builder, different for William
    :type cache_file: callable
    :return: None
    """
    # Adapted from source 1 (cell 22)
    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / str(game_id)[:4]
        # Create the cache folder (parents=True also creates William's <year> subfolder)
        cache_file(folder, game_id).parent.mkdir(parents=True, exist_ok=True)  # William
        # Fill the cache ourselves before the call
        cache_file(folder, game_id).write_text(json.dumps(game), encoding="utf-8")  # William
        with patch(
            "requests.get",
            side_effect=AssertionError(
                "File in cache was ignored : request for an existing file (in cache) was sent"
            ),
        ) as get:
            result = fetch_fn(game_id, folder)
        # If get was called, the function ignored the cache
        get.assert_not_called()
        assert result == game


def check_failed_request(fetch_fn, game_id, cache_file=lambda folder, gid: folder / f"{gid}.json"):
    """Test 3 (BONUS): the request fails (HTTP 404), an error is raised and nothing is cached.

    :param fetch_fn: Function to test
    :type fetch_fn: callable
    :param game_id: Game id
    :type game_id: int
    :param cache_file: Cache path builder, different for William
    :type cache_file: callable
    :return: None
    """
    # Case suggested by source 1 (cell 23)
    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / str(game_id)[:4]
        # Fake API response with Mock
        fake_response = Mock()
        fake_response.status_code = 404  # William : reads status_code instead of raise_for_status
        # Simulate an error response from the API
        fake_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Client Error"
        )
        with patch("requests.get", return_value=fake_response) as get:
            # Try the function
            try:
                fetch_fn(game_id, folder)
            # Pass if it behaves as expected
            except requests.HTTPError:
                pass
            # Otherwise report the problem
            else:
                raise AssertionError("No error even with HTTP flag")
        get.assert_called_once()
        assert not cache_file(folder, game_id).exists(), (  # William
                "File cached even if the client returned an error"
        )


def check_wrong_cache(fetch_fn, game_id, game):
    """Test 4 (BONUS): the cached game is the wrong one, so a ValueError is raised.

    :param fetch_fn: Function to test
    :type fetch_fn: callable
    :param game_id: Game id
    :type game_id: int
    :param game: Reference game
    :type game: dict
    :return: None
    """
    # Case suggested by source 1 (cell 23)
    other_id = game_id + 1
    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / str(game_id)[:4]
        folder.mkdir()
        # Fill the cache ourselves with game_id under the wrong name other_id
        (folder / f"{other_id}.json").write_text(json.dumps(game), encoding="utf-8")
        with patch(
            "requests.get",
            side_effect=AssertionError("Request sent without name error flag"),
        ) as get:
            # Try the function
            try:
                fetch_fn(other_id, folder)
            # Pass if it behaves as expected
            except ValueError:  # To validate with William's code
                pass
            # Otherwise report the problem
            else:
                raise AssertionError(
                    f"cache file {other_id}.json contained a game whose id is {game_id}; the function returned it instead of raising ValueError"
                )
        get.assert_not_called()

# The reasoning behind the season test functions was developed with source 3, to help validate the structure
# of the fake data.

def check_season_download(fetch_season_fn):
    """Regular season: fake fetch_game returns 3 games then a 404; check the loop stops and returns 3 sorted games.

    :param fetch_season_fn: Function to test
    :type fetch_season_fn: callable
    :return: None
    """
    season = 2016
    existing = [2016020001, 2016020002, 2016020003]  # Match in fake season
    calls = []  # list for requested game_id

    # Fake fetch_game because we want to test fetch_season and not receive errors from real fetch_game
    def fake_fetch_game(game_id, cache_dir):
        calls.append(game_id)
        if game_id in existing:
            return {"id": game_id, "plays": []}
        raise http_error(404)

    with TemporaryDirectory() as temp_dir:
        # Code by source 3 to replace the real fetch_game by the fake one in fetch_season
        with patch.dict(
            fetch_season_fn.__globals__, {"fetch_game_llm": fake_fetch_game}
        ):
            result = fetch_season_fn(season, 2, Path(temp_dir))

    # Check if 4 calls and no more because of 404
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
    """Playoffs: check the 8-4-2-1 matchups rule, the jump to the next series on 404 and no extra calls.

    :param fetch_season_fn: Function to test
    :type fetch_season_fn: callable
    :return: None
    """
    matchups = {1: 8, 2: 4, 3: 2, 4: 1}
    # Match in fake serie with proper matchup structure. Fake match was made by CLAUDE for accuracy
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
        raise http_error(404)

    with TemporaryDirectory() as temp_dir:
        # Code by CLAUDE to replace the real fetch_game by the fake one in fetch_season
        with patch.dict(
            fetch_season_fn.__globals__, {"fetch_game_llm": fake_fetch_game}
        ):
            result = fetch_season_fn(2016, 3, Path(temp_dir))
    # Every ID must have 10 numbers
    bad = [i for i in calls if len(str(i)) != 10]
    assert not bad, f"Every game_id must have exactly 10 digits (e.g. 2016030123). These ids are wrong: {bad[:3]}"
    # unexisting serie must not be called
    wrong = [i for i in calls if i in forbidden]
    assert not wrong, (
        f"Rounds 1, 2, 3, 4 have 8, 4, 2, 1 matchups, but these non-existing series were requested: {wrong}"
    )
    # 15 match, function visits each series
    # Added first calls made so refiner can see mistake between good and bad number
    # Changed prompt because it seems to confuse qwen
    ids = [g["id"] for g in result]
    assert ids == existing, (
        f"Some playoff games were not returned: after a 404, the function must move to the next matchup, not stop. Got {len(ids)} games, first calls: {calls[:3]}"
    )


def check_error_prop(fetch_season_fn):
    """Check that a non-404 error (HTTP 500) propagates and stops the loop.

    :param fetch_season_fn: Function to test
    :type fetch_season_fn: callable
    :return: None
    """
    calls = []  # list for requested game_id

    # Fake fetch_game because we want to test fetch_season and not receive errors from real fetch_game
    def fake_fetch_game(game_id, cache_dir):
        calls.append(game_id)
        if game_id == 2016020001:
            return {"id": game_id, "plays": []}
        raise http_error(500)

    with TemporaryDirectory() as temp_dir:
        # Code by CLAUDE to replace the real fetch_game by the fake one in fetch_season
        with patch.dict(
            fetch_season_fn.__globals__, {"fetch_game_llm": fake_fetch_game}
        ):
            try:  # Try function
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
    """Check that a non-existing season returns [] for regular season and playoffs.

    :param fetch_season_fn: Function to test
    :type fetch_season_fn: callable
    :return: None
    """
    calls = []  # list for requested game_id

    # Fake fetch_game because we want to test fetch_season and not receive errors from real fetch_game
    def fake_fetch_game(game_id, cache_dir):
        calls.append(game_id)
        raise http_error(404)

    with TemporaryDirectory() as temp_dir:
        # Code by CLAUDE to replace the real fetch_game by the fake one in fetch_season
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
    """Integration test with the real fetch_game: cached games are not requested again.

    :param fetch_season_fn: Function to test
    :type fetch_season_fn: callable
    :return: None
    """
    cached = [2016020001, 2016020002]
    url = "https://api-web.nhle.com/v1/gamecenter/2016020003/play-by-play"

    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir) / str(game_id)[:4]
        folder.mkdir()
        # we fill cache with 2 valid match
        for game_id in cached:
            (folder / f"{game_id}.json").write_text(
                json.dumps({"id": game_id, "plays": []}), encoding="utf-8"
            )
        # Fake server response with mock
        fake_response = Mock()
        fake_response.raise_for_status.side_effect = http_error(404)
        with patch("requests.get", return_value=fake_response) as get:
            result = fetch_season_fn(2016, 2, folder)

        get.assert_called_once_with(url, timeout=30)
        assert not (folder / "2016020003.json").exists(), (
            "File cached for a game that returned 404"
        )
        ids = [g["id"] for g in result]
        assert ids == cached, f"Expected the 2 cached games {cached}, got: {ids}"

def run_tests(fetch_fn, game_id, game, cache_file=lambda folder, gid: folder / f"{gid}.json"):
    """Run the 4 game tests on fetch_fn.

    :param fetch_fn: Function to test
    :type fetch_fn: callable
    :param game_id: Game id
    :type game_id: int
    :param game: Reference game
    :type game: dict
    :param cache_file: Cache path builder, different for William
    :type cache_file: callable
    :return: {test name: status}
    :rtype: dict
    """
    # Dictionary so we can loop over our functions. lambda idea comes from CLAUDE
    # lambda creates a function without name or parameter that runs the test when it is called
    test = {
        "1. Known game": lambda: check_download(fetch_fn, game_id, game, cache_file),
        "2. Cache check": lambda: check_cache_no_request(
            fetch_fn, game_id, game, cache_file
        ),
        "3. Failed HTTP request": lambda: check_failed_request(
            fetch_fn, game_id, cache_file
        ),
        "4. Wrong game in cache": lambda: check_wrong_cache(
            fetch_fn, game_id, game
        ),
    }
    results = {}
    for name, test in test.items():
        try:
            test()  # Test function is run
            results[name] = "OK"
        except AssertionError as e:  # If test function failed. Before Exception because assertion is exception
            results[name] = f"FAIL : {e}"
        except Exception as e:  # Recommended by CLAUDE, allows error identification other than my Assert
            results[name] = (
                f"ERROR : {type(e).__name__}: {e}"  # Prints the Exception type
            )
    return results

def run_season_tests(fetch_season_fn):
    """Run the 5 season tests on fetch_season_fn. Same structure as run_tests.

    :param fetch_season_fn: Function to test
    :type fetch_season_fn: callable
    :return: {test name: status}
    :rtype: dict
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
            results[name] = f"FAIL : {e}"
        except Exception as e:
            results[name] = f"ERROR : {type(e).__name__}: {e}"
    return results
