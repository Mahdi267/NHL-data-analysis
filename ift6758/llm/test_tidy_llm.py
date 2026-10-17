import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
GAME_FILE = RAW_DIR / "2022030411.json"
COLUMNS = ["game_id", "event_id", "season", "game_type", "period", "time", "team_id",
           "event_type", "x_coord", "y_coord", "shooter", "goalie", "shot_type",
           "empty_net", "strength"]


def _load_game():
    """Loads game to comapre with results

    """
    return json.loads(GAME_FILE.read_text(encoding="utf-8"))


def check_structure(tidy_fn):
    """TChecks if retrun is pandas with right columns like Madhi's code

    """
    df = tidy_fn(str(GAME_FILE))
    assert isinstance(df, pd.DataFrame), f"Must return a pandas DataFrame, got: {type(df).__name__}"
    missing = [c for c in COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}. Columns returned: {list(df.columns)}"
    assert len(df) > 0, "The DataFrame is empty for a real game"

def run_tidy_tests(tidy_fn):
    """Run the  generic tidy tests on tidy_fn. Same structure as run_tests."""
    tests = {
        "1. Structure": lambda: check_structure(tidy_fn),
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