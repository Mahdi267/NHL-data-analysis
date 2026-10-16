#Functions used for Task 3: Simple visualizations
from ift6758.features import json2tidy
from pathlib import Path
import pandas as pd
import numpy as np

def games_ids(season):
    """
    Return in a list all the game ids for the season.

    Parameters:
            Season (int) : Season to take the games from

    """
    pathlist = Path(f"../notebooks/data/raw/{season}")
    return [p.stem for p in pathlist.glob("*.json")]

def games_list(season, events=True) -> pd.DataFrame:
    """
    Function to put all the games dataframe for a season in a list and merge them into one dataframe.

    Parameters:
        Season (int) : Season to take the games from

    Returns:
        pd.DataFrame
    """
    df_games_list = []

    for game_id in games_ids(season):
        df = json2tidy(f"../notebooks/data/raw/{season}/{game_id}.json", events)
        df_games_list.append(df)

    merged_df_regular = pd.concat(df_games_list, ignore_index=True)

    return merged_df_regular


def distance(dataframe) -> pd.DataFrame:
    """
    Function to add the column distance to the dataframe.
    Distance is the distance between the goal and the coordinates where the shot was taken

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
    df["distance"] = np.sqrt((df["goalCoordX"]-df["x_coord"])**2 + df["y_coord"]**2)

    #Remove team_defending, side_defending, goalCoordX
    #Keep only the distance
    df = df.drop(
        columns = ["team_defending", "side_defending", "goalCoordX"]
    )

    return df
