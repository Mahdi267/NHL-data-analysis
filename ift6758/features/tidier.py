import json
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd


# Constants
GOAL_X = 89                                 # x coordinate of the goal line (feet)
PERIOD_SECONDS = 20 * 60                    # length of a regulation period (seconds)
SHOT_EVENTS = ["shot-on-goal", "goal"]      # events kept in the final dataset
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def game_to_events(game: dict):
    """
    Convert one raw play-by-play game into a DataFrame with ALL events. Derived form json2tidy from milestone 1

    Parameters:
        game (dict) : Raw JSON of one game (api-web.nhle.com/v1/gamecenter/{GAME_ID}/play-by-play)

    Returns:
        pd.DataFrame : One row per event, sorted by sort_order
    """
    df_data = pd.json_normalize(
        game,
        "plays",
        [
            "id",
            "season",
            "gameType",
            ["homeTeam", "id"],
            ["homeTeam", "abbrev"],
            ["awayTeam", "id"],
            ["awayTeam", "abbrev"],
        ],
    )
    new_columns = {"id":"game_id",
              "eventId":"event_id",
              "season":"season",
              "sortOrder":"sort_order",
              "gameType":"game_type",
              "periodDescriptor.number":"period",
              "periodDescriptor.periodType":"period_type",
              "timeInPeriod":"time_in_period",
              "typeDescKey":"event_type",
              "details.xCoord":"x_coord",
              "details.yCoord":"y_coord",
              "details.shotType":"shot_type",
              "details.eventOwnerTeamId":"team_id",
              "homeTeam.id":"home_team_id",
              "homeTeamDefendingSide":"home_defending_side",
              "situationCode":"situation_code",
              "details.typeCode":"penalty_type",
              "details.duration":"penalty_duration"}
    df_data = (df_data.reindex(columns=new_columns.keys())).rename(columns=new_columns)
    df_data = fill_home_defending_side(df_data)
    df_data = add_game_seconds(df_data)
    df_data = add_shot_geometry(df_data)
    # To help with time inconsistency (seconds are sometimes not in the right order) we tried ordering only by time.
    # Claude recommended to use both seconds AND sort_values to sort with quicksort or to use stable or mergesort as kind
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
    df_data = (df_data.sort_values(by=["game_seconds","sort_order"])).reset_index(drop=True) # To help with Q4.2

    return df_data


def add_game_seconds(dataframe: pd.DataFrame):
    """
    Add the column game_seconds: seconds elapsed since the start of the game.
    IMPORTANT: Sometimes, sort-order is not aligned with timne if seconds elapse between note taking.

    Parameters:
        dataframe (pd.DataFrame) : Output of game_to_events (needs period and time_in_period)

    Returns:
        pd.DataFrame
    """
    data = dataframe.copy()
    time_str = data["time_in_period"].str.split(":",expand=True)
    seconds = pd.to_numeric(data["period"]-1)*1200 + pd.to_numeric(time_str[0])*60+pd.to_numeric(time_str[1])
    data["game_seconds"] = seconds
    return data


def fill_home_defending_side(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates homeTeamDefendingSide information if missing

    param:
        dataframe (pd.DataFrame) : Output of game_to_events (needs period and time_in_period)

    return:
        pd.DataFrame
    """
    data = dataframe.copy()
    if data["home_defending_side"].notna().any(): # If info exist, we skip estimation
        data["home_side_inferred"] = False # We didnt estimate
        return data
    shots = data[data["event_type"].isin(SHOT_EVENTS) & (data["period_type"] != "SO")] # fusillade -> tous les tirs dans le même net
    shots = shots.dropna(subset=["x_coord"])
    x = np.where(shots["team_id"] == shots["home_team_id"], shots["x_coord"], -shots["x_coord"]) # Make sure we mesure median for one team and not cancel by calculing both
    score = shots.assign(x_home=x)
    score = score.groupby(["period"])["x_home"].median()
    
    # score > 0: home team shoots right and defend left
    # if score > 0:
    #     side = "left"
    # elif score < 0:
    #     side = "right"
    # else:
    #     side = None
    # nupy method that replicates if/elif/else more efficiently https://numpy.org/doc/2.4/reference/generated/numpy.select.html
    # array = np.select([score > 0, score < 0], ["left", "right"], default=None)
    # We put it back into a indexed pd.Series
    #arena_side = pd.Series(array,index=score.index)

    # score > 0: home team shoots right and defend left
    arena_side = np.sign(score).map({1:"left",-1:"right"})
    data["home_defending_side"] =  data["period"].map(arena_side)
    data["home_side_inferred"] = True # data was estimated
    return data
    

def add_shot_geometry(dataframe: pd.DataFrame):
    """
    Add the columns distance and angle relative to the net attacked by the event owner team.
    Unknown defending side gives NaN (not a default side).

    param:
        dataframe (pd.DataFrame) : Needs x_coord, y_coord, team_id, home_team_id, home_defending_side

    returns:
        pd.DataFrame with data["distance"] in feet, data["angle"] in degrees
    """

    data = dataframe.copy()
    # Adapted from Williams question 2 code

    # Shot direction for arctan2
    home_direction = data["home_defending_side"].map({"left":1,"right":-1})
    # If player isnt part of home team, flip direction, all wrapped in a pdseries.
    s = pd.Series(home_direction.where(data["team_id"] == data["home_team_id"], - home_direction))

    dx = GOAL_X - s * data["x_coord"]

    data["distance"] = np.hypot(dx,data["y_coord"])

    # We use arctan for the full 180 to -180 (behnd the goal) information. Angle in degress, 0 to 180
    data["angle"] = np.degrees(np.arctan2(np.abs(data["y_coord"]), dx))

    return data


def filter_shots(dataframe: pd.DataFrame, drop_shootout: bool = True):
    """
    Keep only shot-on-goal and goal events.

    param:
        dataframe (pd.DataFrame) : Events of one or several games
        drop_shootout (bool) : If True, remove shootout attempts (period_type == "SO")

    return:
        pd.DataFrame
    """
    data = dataframe.copy()
    if drop_shootout:
        return data[data["event_type"].isin(SHOT_EVENTS) & (data["period_type"] != "SO")].reset_index(drop=True)

    return data[data["event_type"].isin(SHOT_EVENTS)].reset_index(drop=True)


def build_dataset(seasons: list[int], data_dir: Path = DEFAULT_DATA_DIR):
    """
    Apply the full pipeline to every regular season game of the given seasons.

    param:
        seasons (list[int]) : First year of each season, e.g. [2016, 2017]
        data_dir (Path) : Folder containing one sub-folder per season with the raw JSON files

    retunr:
        pd.DataFrame : All shots and goals of the given seasons
    """
    frames = []
    for season in seasons:
        files = sorted(Path(data_dir, str(season)).glob(f"{season}02*.json"))
        for path in tqdm(files, desc=str(season)):
            with open(path, encoding="utf-8") as f:
                events = game_to_events(json.load(f))
            frames.append(filter_shots(events))
    dataset = pd.concat(frames,ignore_index=True)
    return dataset
