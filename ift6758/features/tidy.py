# references:
# https://pandas.pydata.org/docs
# https://www.geeksforgeeks.org/pandas/
# https://www.youtube.com/watch?v=YZp9hxwP7R4&t=21s

import pandas as pd
import json


# THIS IS HOW THE DATA IS STRUCTURED IN THE JSON FILE
#
# id : GAME_ID
# season: SEASON_ID
# ...
# awayTeam: {id: TEAM_ID, abbrev: "ABBREV", score: SCORE, ...}
# homeTeam: {id: TEAM_ID, ...}
# ...
# plays: [{...}, {...}, ...]
# ...
# rosterSpots: [{...}, {...}, ...] -> players information
#
# For each play (shots and goals only):
# {
#     eventId: EVENT_ID,
#     periodDescriptor: {
#         number,
#         periodType,
#         maxRegulationPeriods
#     },
#     timeInPeriod,
#     timeRemaining,
#     situationCode,
#     homeTeamDefendingSide,
#     typeCode,
#     typeDescKey,
#     sortedOrder,
#     details: {
#         xCoord,
#         yCoord,
#         zoneCode,
#         shotType,
#         shootingPlayerId,
#         goalieInNetId,
#         eventOwnerTeamId,
#         awaySOG,
#         homeSOG
#     }
# }


# We are looking on creating the dataframe that contains this information:
# - game ID
# - event ID
# - season
# - game type
# - period and time in period
# - team ID of the team that took the shot
# - event type (shot or goal)
# - x and y coordinates of the shot
# - shooting player name
# - goalie name
# - shot type
# - whether the shot was taken on an empty net
# - whether the goal was scored at even strength, shorthanded, or on the power play
def json2tidy(raw_data: str | dict) -> pd.DataFrame:
    """Convert raw NHL play-by-play JSON data into a tidy DataFrame."""

    if isinstance(raw_data, str):
        with open(raw_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = raw_data

    # Normalize the json data
    df_data = pd.json_normalize(data, "plays", ["id", "season"])

    # Filter the goal and shot-on-goal only
    df_goals_shots = df_data[
        df_data["typeDescKey"].isin(["goal", "shot-on-goal"])
    ].copy()

    # Extract the game type from the game ID
    game_type_code = str(data["id"])[4:6]

    if game_type_code == "02":
        game_type = "regular season"
    elif game_type_code == "03":
        game_type = "playoffs"
    else:
        game_type = None

    wanted_columns = [
        "id",
        "eventId",
        "season",
        "situationCode",
        "periodDescriptor.number",
        "timeInPeriod",
        "details.shootingPlayerId",
        "details.scoringPlayerId",
        "typeDescKey",
        "details.xCoord",
        "details.yCoord",
        "details.goalieInNetId",
        "details.shotType",
        "details.eventOwnerTeamId"
    ]

    # Select only wanted information
    df_filtered_info = df_goals_shots.reindex(columns=wanted_columns).copy()
    df_filtered_info["game_type"] = game_type

    # Normalize the json data to get players name
    df_name_data = pd.json_normalize(data, "rosterSpots")

    # Keep first and last name only
    df_name_data = df_name_data.loc[:, [
        "playerId",
        "firstName.default",
        "lastName.default"
    ]]

    # Merge name information for each type of action

    # Shooters name
    df_merged_name_play_shooter = pd.merge(
        df_filtered_info,
        df_name_data,
        left_on="details.shootingPlayerId",
        right_on="playerId",
        how="left"
    ).rename(columns={
        "firstName.default": "shooterName",
        "lastName.default": "shooterLastName"
    })

    # Remove the player ID used for the first merge
    df_merged_name_play_shooter = df_merged_name_play_shooter.drop(
        columns=["playerId"]
    )

    # Scorers name
    df_merged_name_play_scorer = pd.merge(
        df_merged_name_play_shooter,
        df_name_data,
        left_on="details.scoringPlayerId",
        right_on="playerId",
        how="left"
    ).rename(columns={
        "firstName.default": "scorerName",
        "lastName.default": "scorerLastName"
    })

    # Remove the player ID used for the second merge
    df_merged_name_play_scorer = df_merged_name_play_scorer.drop(
        columns=["playerId"]
    )

    # Goalies name
    df_merged_name_play = pd.merge(
        df_merged_name_play_scorer,
        df_name_data,
        left_on="details.goalieInNetId",
        right_on="playerId",
        how="left"
    ).rename(columns={
        "firstName.default": "goalieName",
        "lastName.default": "goalieLastName"
    })

    # Remove the player ID used for the third merge
    df_merged_name_play = df_merged_name_play.drop(
        columns=["playerId"]
    )

    # Add a column to check if net was empty
    # goalieInNetId is missing when the goalie has been pulled
    df_merged_name_play["emptyNet"] = (
        df_merged_name_play["details.goalieInNetId"].isna()
    )

    # Extract homeTeam and awayTeam info
    df_home_away = pd.json_normalize(data).loc[:, [
        "homeTeam.id",
        "awayTeam.id"
    ]]

    home_id = df_home_away.loc[:, "homeTeam.id"].values[0]
    away_id = df_home_away.loc[:, "awayTeam.id"].values[0]

    # Add a column for the strength of the goal
    df_merged_name_play["strength"] = None

    # Map situation code to explicit value
    # The strength information is only available for goals

    # The situation code is structured as:
    # away goalie - away skaters - home skaters - home goalie
    #
    # 1551 = 5 vs 5
    # 1541 = away 5 vs home 4
    # 1451 = away 4 vs home 5

    # 1. Even strength
    df_merged_name_play.loc[
        (df_merged_name_play["typeDescKey"] == "goal") &
        (
            df_merged_name_play["situationCode"].str[1].astype(int) ==
            df_merged_name_play["situationCode"].str[2].astype(int)
        ),
        "strength"
    ] = "even"

    # 2. Home team is on the power play
    df_merged_name_play.loc[
        (df_merged_name_play["typeDescKey"] == "goal") &
        (
            df_merged_name_play["situationCode"].str[1].astype(int) >
            df_merged_name_play["situationCode"].str[2].astype(int)
        ) &
        (df_merged_name_play["details.eventOwnerTeamId"] == home_id),
        "strength"
    ] = "power play"

    # 3. Away team is on the power play
    df_merged_name_play.loc[
        (df_merged_name_play["typeDescKey"] == "goal") &
        (
            df_merged_name_play["situationCode"].str[2].astype(int) >
            df_merged_name_play["situationCode"].str[1].astype(int)
        ) &
        (df_merged_name_play["details.eventOwnerTeamId"] == away_id),
        "strength"
    ] = "power play"

    # 4. Home team is shorthanded
    df_merged_name_play.loc[
        (df_merged_name_play["typeDescKey"] == "goal") &
        (
            df_merged_name_play["situationCode"].str[1].astype(int) >
            df_merged_name_play["situationCode"].str[2].astype(int)
        ) &
        (df_merged_name_play["details.eventOwnerTeamId"] == away_id),
        "strength"
    ] = "shorthanded"

    # 5. Away team is shorthanded
    df_merged_name_play.loc[
        (df_merged_name_play["typeDescKey"] == "goal") &
        (
            df_merged_name_play["situationCode"].str[2].astype(int) >
            df_merged_name_play["situationCode"].str[1].astype(int)
        ) &
        (df_merged_name_play["details.eventOwnerTeamId"] == home_id),
        "strength"
    ] = "shorthanded"

    # Rename columns to make the dataframe easier to read
    df_merged_name_play = df_merged_name_play.rename(columns={
        "id": "game_id",
        "eventId": "event_id",
        "season": "season",
        "situationCode": "situation_code",
        "periodDescriptor.number": "period",
        "timeInPeriod": "time",
        "details.shootingPlayerId": "shooter_id",
        "details.scoringPlayerId": "scorer_id",
        "typeDescKey": "event_type",
        "details.xCoord": "x_coord",
        "details.yCoord": "y_coord",
        "details.goalieInNetId": "goalie_id",
        "details.shotType": "shot_type",
        "details.eventOwnerTeamId": "team_id",
        "shooterName": "shooter",
        "shooterLastName": "shooter_last_name",
        "scorerName": "scorer",
        "scorerLastName": "scorer_last_name",
        "goalieName": "goalie",
        "goalieLastName": "goalie_last_name"
    })

    return df_merged_name_play
    
if __name__ == "__main__":
    print(json2tidy("ift6758/notebooks/data/raw/2022/2022020001.json")["strength"])