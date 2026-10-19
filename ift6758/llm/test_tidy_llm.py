"""
Tests used to compare the Qwen-generated tidy function with Mahdi's json2tidy (ift6758/features/tidy.py).
Same structure as test_both_code.py.
"""
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("../data")
GAME_FILE = RAW_DIR / "2022030411.json"
COLUMNS = ["game_id", "event_id", "season", "game_type", "period", "time", "team_id",
           "event_type", "x_coord", "y_coord", "shooter", "goalie", "shot_type",
           "empty_net", "strength"]
KEY = ["game_id", "event_id"]
NUMERIC = ["game_id", "event_id", "season", "game_type", "period", "team_id", "x_coord", "y_coord"] # Numeric value, used in normalised
TEXT = ["time", "event_type", "shooter", "goalie", "shot_type", "strength"]


def _load_game():
    """Load the reference game to compare with the results.

    :return: Game JSON
    :rtype: dict
    """
    return json.loads(GAME_FILE.read_text(encoding="utf-8"))


def check_structure(tidy_fn):
    """Check that tidy_fn returns a non-empty DataFrame with the same columns as Mahdi's code.

    :param tidy_fn: Function to test
    :type tidy_fn: callable
    :return: None
    """
    df = tidy_fn(str(GAME_FILE))
    assert isinstance(df, pd.DataFrame), f"Must return a pandas DataFrame, got: {type(df).__name__}"
    missing = [c for c in COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}. Columns returned: {list(df.columns)}"
    assert len(df) > 0, "The DataFrame is empty for a real game"

def check_rows(tidy_fn):
    """Check that there is one row per shot on goal or goal of the JSON, with a unique (game_id, event_id).

    :param tidy_fn: Function to test
    :type tidy_fn: callable
    :return: None
    """
    game = _load_game()
    df = tidy_fn(str(GAME_FILE))
    # On compte nous-mêmes les tirs cadrés et les buts dans le JSON
    expected = sum(p["typeDescKey"] in ("shot-on-goal", "goal") for p in game["plays"])
    assert len(df) == expected, f"Expected {expected} rows (shots on goal + goals in the JSON), got {len(df)}"
    assert set(df["event_type"]) <= {"shot-on-goal", "goal"}, (
        f"event_type must be shot-on-goal or goal, got: {set(df['event_type'])}"
    )
    assert not df.duplicated(["game_id", "event_id"]).any(), "Some (game_id, event_id) appear more than once"
    assert set(df["game_id"]) == {game["id"]}, f"game_id must be {game['id']}, got: {set(df['game_id'])}"


def check_values(tidy_fn):
    """Check that the values make sense: teams of the game, coordinates on the ice,
    a shooter for every event, empty_net True/False and strength only for goals.

    :param tidy_fn: Function to test
    :type tidy_fn: callable
    :return: None
    """
    game = _load_game()
    df = tidy_fn(str(GAME_FILE))
    teams = {game["homeTeam"]["id"], game["awayTeam"]["id"]}
    other_teams = set(df["team_id"].dropna()) - teams
    assert not other_teams, f"team_id must be one of the two teams {teams}, found: {other_teams}"
    # Patinoire LNH : 200 x 85 pieds, centre = (0, 0)
    outside = df[(df["x_coord"].abs() > 100) | (df["y_coord"].abs() > 42.5)]
    assert outside.empty, f"{len(outside)} events are outside the rink (|x| <= 100, |y| <= 42.5)"
    no_shooter = df[df["shooter"].isna()]
    assert no_shooter.empty, (
        f"Every shot and goal has a shooter, but {len(no_shooter)} rows have none "
        f"({no_shooter['event_type'].value_counts().to_dict()})"
    )
    assert set(df["empty_net"]) <= {True, False}, f"empty_net must be True or False, got: {set(df['empty_net'])}"
    goals = df[df["event_type"] == "goal"]
    shots = df[df["event_type"] == "shot-on-goal"]
    assert set(goals["strength"]) <= {"even", "shorthanded", "power play"}, (
        f"Goal strength must be even, shorthanded or power play, got: {set(goals['strength'])}"
    )
    assert shots["strength"].isna().all(), "strength must be missing for shots"


# Function made by Claude
def _normalize(df):
    """Put both DataFrames in the same types so that 13 == 13.0 and NaN == None.

    :param df: DataFrame to normalize
    :type df: pd.DataFrame
    :return: Normalized copy
    :rtype: pd.DataFrame
    """
    out = df.copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in TEXT:
        out[col] = out[col].where(out[col].notna(), None).astype(object)
    out["empty_net"] = out["empty_net"].map({True: True, False: False, "True": True, "False": False})
    return out

# Function made by Claude
def compare_to_reference(tidy_fn, reference, games):
    """Run tidy_fn on each game and compare its DataFrame with the reference, column by column.

    :param tidy_fn: Function to compare (Qwen or Mahdi)
    :type tidy_fn: callable
    :param reference: Reference DataFrame (reference_tidy.csv)
    :type reference: pd.DataFrame
    :param games: {game_id: path to the game JSON}
    :type games: dict
    :return: {"rows": {...}, "agreement_%": {column: %}, "mismatch_examples": {column: [(game_id, event_id), ...]}}
    :rtype: dict
    """
    columns = [c for c in reference.columns if c not in KEY]
    llm = pd.concat([tidy_fn(str(path)) for path in games.values()], ignore_index=True)
    llm = _normalize(llm[KEY + columns])
    ref = _normalize(reference[reference["game_id"].isin(games)])

    # outer + indicator : on voit les lignes manquantes (left_only) et en trop (right_only)
    merged = ref.merge(llm, on=KEY, how="outer", suffixes=("_ref", "_llm"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    examples = {}
    rows = {
        "reference": len(ref),
        "llm": len(llm),
        "matched": len(both),
        "missing": int((merged["_merge"] == "left_only").sum()),
        "extra": int((merged["_merge"] == "right_only").sum()),
    }
    # (game_id, event_id) des lignes en trop / manquantes, pour comprendre la différence
    for kind, flag in (("missing", "left_only"), ("extra", "right_only")):
        if rows[kind]:
            examples[f"{kind} rows"] = list(
                merged.loc[merged["_merge"] == flag, KEY].astype(int).itertuples(index=False, name=None)
            )[:5]

    agreement = {}
    for col in columns:
        a, b = both[f"{col}_ref"], both[f"{col}_llm"]
        same = (a == b) | (a.isna() & b.isna())  # NaN == NaN compte comme identique
        label = col
        agreement[label] = round(100 * same.mean(), 1) if len(both) else 0.0
        if not same.all():
            examples[col] = list(both.loc[~same, KEY].astype(int).itertuples(index=False, name=None))[:5]
    return {"rows": rows, "agreement_%": agreement, "mismatch_examples": examples}

def add_goalie_stops(df):
    """Add the column goalie_stops to a tidy DataFrame of shots and goals.

    Used William (or Tiffani's) description of the caracteristic as rules.
    For each game and each goalie, the counter adds 1 for every shot on goal and 0 for every goal.
    The first event of a goalie starts at 1 (shot on goal) or 0 (goal). 
    :type df: pd.DataFrame
    :return: Copy of df with the new column goalie_stops (missing when the net is empty)
    :rtype: pd.DataFrame
    """
    out = df.copy()
    # 1 for a stopped shot, 0 for a goal
    stopped = (out["event_type"] == "shot-on-goal").astype(int)
    # Running total per game and per goalie, in the order of the rows.
    # Rows without goalie (empty net) are left out by groupby and stay missing.
    out["goalie_stops"] = stopped.groupby([out["game_id"], out["goalie"]]).cumsum()
    out["goalie_stops"] = out["goalie_stops"].astype("Int64")  # Integer that allows missing values
    return out

def run_tidy_tests(tidy_fn):
    """Run the generic tidy tests on tidy_fn. Same structure as run_tests.

    :param tidy_fn: Function to test
    :type tidy_fn: callable
    :return: {test name: status}
    :rtype: dict
    """
    tests = {
        "1. Structure": lambda: check_structure(tidy_fn),
        "2. Rows": lambda: check_rows(tidy_fn),
        "3. Values format": lambda: check_values(tidy_fn),
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
