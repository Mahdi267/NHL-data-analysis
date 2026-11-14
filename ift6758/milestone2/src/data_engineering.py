"""
Data engineering for Q2 and Q4.

Source: Milestone 1, William Perktold, ift6758/notebooks/raw_data_acquisition.ipynb
"""
import json
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from ift6758.milestone2.src.data_extraction import NHLyear, DATA_DIR, games_list


# Constants
GOAL_X = 89  # x coordinate of the goal line (feet)
PERIOD_SECONDS = 20 * 60  # length of a regulation period (seconds)
SHOT_EVENTS = ["shot-on-goal", "goal"]  # events asked in README

def build_dataset_q2(seasons: list, data_dir: Path = DATA_DIR, game_type: str = "02") -> pd.DataFrame:
    """
    New implementation of raw_data_acquisition.ipynb from Milestone1 mixed with basic_feature_engineering.ipynb
    Function to create dataset from selected season
    
    Parameters:
        seasons (list) : List of seasons
        data_dir (Path) : Path to take season from
        game_type (str) : "02" for regular season or "03" for series
        
    Returns:
        tidy (pd.Dataframe):
    """
    tidy = []
    for season in seasons:
        if not any(Path(data_dir, str(season)).glob(f"{season}{game_type}*.json")):
            yr = NHLyear(season, data_dir)  # Creation of the object
            yr.download_all()  # downloads all the games found
        tidy.append(games_list(season,data_dir=data_dir, game_type=game_type))

    tidy = pd.concat(tidy,ignore_index=True)
    tidy = angles_will(tidy)
    tidy.replace(np.inf, np.nan, inplace=True)
    tidy.dropna(subset=["distance", "angle"], inplace=True)
    return tidy

def angles_will(dataframe, ) -> pd.DataFrame:
    """
    Modified version of "distance" function from Milestone 1
    Function to add the columns distance and angle to the dataframe.
    Distance is the distance between the goal and the coordinates where the shot was taken
    Angle is the angle of the shot relative to the goal

    Parameters:
        dataframe (pd.DataFrame) : Dataframe of the games

    Returns:
        pd.DataFrame
    """
    df = dataframe.copy()
    # team_defending: True is home_team, False is away_team
    # df["team_id"] contain "eventOwnerTeamId" which is the team associated with the action
    # df["team_id"] is the team that is trying to score
    df["team_defending"] = (df["team_id"] != df["homeTeam_id"])

    df["side_defending"] = df["homeTeamDefendingSide"]

    conditions = [df["side_defending"]=="right",df["side_defending"]=="left"]
    choices = ["left","right"]

    df["side_defending"] = np.where(df["team_defending"]==False,np.select(conditions,choices),df["side_defending"])

    # if right, goal_coor = (89,0)
    # else left, (-89,0)
    df["goalCoordX"] = np.where(df["side_defending"]=="right",89,-89)

    #calculate the distance between the x,y coordinates and the goal
    #the y coordinate for the goals is 0
    dx = df["goalCoordX"]-df["x_coord"]


    df["distance"] = np.sqrt((dx)**2 + df["y_coord"]**2)

    #df["angle"] = np.arctan(df["y_coord"]/(df["goalCoordX"]-df["x_coord"]))
    df["angle"] = np.arctan(df["y_coord"]/(dx))

    #Remove team_defending, side_defending, goalCoordX
    #Keep only the distance
    df = df.drop(
        columns = ["team_defending", "side_defending", "goalCoordX"]
    )
    return df


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
    new_columns = {
        "id": "game_id",
        "eventId": "event_id",
        "season": "season",
        "sortOrder": "sort_order",
        "gameType": "game_type",
        "periodDescriptor.number": "period",
        "periodDescriptor.periodType": "period_type",
        "timeInPeriod": "time_in_period",
        "typeDescKey": "event_type",
        "details.xCoord": "x_coord",
        "details.yCoord": "y_coord",
        "details.shotType": "shot_type",
        "details.eventOwnerTeamId": "team_id",
        "homeTeam.id": "home_team_id",
        "homeTeamDefendingSide": "home_defending_side",
        "situationCode": "situation_code",
        "details.typeCode": "penalty_type",
        "details.duration": "penalty_duration",
    }
    df_data = (df_data.reindex(columns=new_columns.keys())).rename(columns=new_columns)

    df_data = fill_home_defending_side(df_data)
    df_data = add_game_seconds(df_data)
    df_data = add_shot_geometry(df_data)
    # To help with time inconsistency (seconds are sometimes not in the right order) we tried ordering only by time.
    # Claude recommended to use both seconds AND sort_values to sort with quicksort or to use stable or mergesort as kind
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
    df_data = (df_data.sort_values(by=["game_seconds", "sort_order"])).reset_index(
        drop=True
    )  # To help with Q4.2
    df_data = last_event(df_data)

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
    time_str = data["time_in_period"].str.split(":", expand=True)
    seconds = (
        pd.to_numeric(data["period"] - 1) * 1200
        + pd.to_numeric(time_str[0]) * 60
        + pd.to_numeric(time_str[1])
    )
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
    if data["home_defending_side"].notna().any():  # If info exist, we skip estimation
        data["home_side_inferred"] = False  # We didnt estimate
        return data
    shots = data[
        data["event_type"].isin(SHOT_EVENTS) & (data["period_type"] != "SO")
    ]  # fusillade -> tous les tirs dans le même net
    shots = shots.dropna(subset=["x_coord"])
    x = np.where(
        shots["team_id"] == shots["home_team_id"], shots["x_coord"], -shots["x_coord"]
    )  # Make sure we mesure median for one team and not cancel by calculing both
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
    # arena_side = pd.Series(array,index=score.index)

    # score > 0: home team shoots right and defend left
    arena_side = np.sign(score).map({1: "left", -1: "right"})
    data["home_defending_side"] = data["period"].map(arena_side)
    data["home_side_inferred"] = True  # data was estimated
    return data


def add_shot_geometry(dataframe: pd.DataFrame):
    """
    Add the columns distance and angle relative to the net attacked by the event owner team.
    Unknown defending side gives NaN.

    param:
        dataframe (pd.DataFrame) : Needs x_coord, y_coord, team_id, home_team_id, home_defending_side

    returns:
        pd.DataFrame with data["distance"] in feet, data["angle"] in degrees
    """

    data = dataframe.copy()
    # Adapted from Williams question 2 code

    # Shot direction for arctan2
    home_direction = data["home_defending_side"].map({"left": 1, "right": -1})
    # If player isnt part of home team, flip direction, all wrapped in a pdseries.
    s = pd.Series(
        home_direction.where(data["team_id"] == data["home_team_id"], -home_direction)
    )

    dx = GOAL_X - s * data["x_coord"]

    data["distance"] = np.hypot(dx, data["y_coord"])

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
        return data[
            data["event_type"].isin(SHOT_EVENTS) & (data["period_type"] != "SO")
        ].reset_index(drop=True)

    return data[data["event_type"].isin(SHOT_EVENTS)].reset_index(drop=True)


def last_event(dataframe: pd.DataFrame):
    """
    Add last event information.

    param:
        dataframe (pd.DataFrame) : Events of one game

    return:
        pd.DataFrame
    """
    # since the data is sorted by seconds and sorted_order, I can use the index
    df0 = dataframe.copy()

    df = dataframe.copy()
    df = df[df.index != 0]
    # Put the first line at the end so the dataframes are the same length
    line0 = df0[df0.index == 0]
    df = pd.concat([df, line0])

    # Last event type
    df["last_event_type"] = df0["event_type"].values
    df.loc[df.index == 0, "last_event_type"] = ""

    # Coordinates of last event
    df["last_x_coord"] = df0["x_coord"].values
    df.loc[df.index == 0, "last_x_coord"] = None
    df["last_y_coord"] = df0["y_coord"].values
    df.loc[df.index == 0, "last_y_coord"] = None

    # Time from last event
    df["time_from_last"] = df["game_seconds"].values - df0["game_seconds"].values
    df.loc[df.index == 0, "time_from_last"] = None

    # Distance from last event
    dx = df["x_coord"].values - df["last_x_coord"].values
    dy = df["y_coord"].values - df["last_y_coord"].values
    df["distance_from_last"] = np.hypot(dx, dy)

    # Rebound
    df["rebound"] = np.where(df["last_event_type"] == "shot-on-goal", True, False)

    # Change in shot angle
    df["change_angle"] = np.where(
        df["rebound"] == True, df["angle"].values - df0["angle"].values, 0
    )

    # Put the first line back on top
    line0 = df[df.index == 0]
    df = df[df.index != 0]
    df = pd.concat([line0, df])

    # Speed
    # When time_from_last==0, distance_from_last=NaN
    #df["speed"] = np.where(
    #    df["time_from_last"] != 0,
    #    df["distance_from_last"].values / df["time_from_last"].values,
    #    df["distance_from_last"],
    #)
    # Pour éviter les messages de divisions par 0, on remplace par
    np.divide(
        df["distance_from_last"].values,
        df["time_from_last"].values,
        out=np.full_like(df["distance_from_last"].values, np.nan, dtype=float), # retourne NaN si faux
        where=(df["time_from_last"].values != 0), # Si temps =0, pas de division
    )

    return df


def build_dataset_q4(seasons: list[int], data_dir: Path = DATA_DIR, game_type: str = "02"):
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
        # Evolution of
        files = sorted(Path(data_dir, str(season)).glob(f"{season}{game_type}*.json"))
        for path in tqdm(files, desc=str(season)):
            with open(path, encoding="utf-8") as f:
                events = game_to_events(json.load(f))
            frames.append(filter_shots(events))
    # ignore_index = True allows to not repeat index for every season. 
    dataset = pd.concat(frames, ignore_index=True)
    return dataset