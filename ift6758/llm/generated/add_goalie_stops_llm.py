import pandas as pd


def add_goalie_stops_llm(df):
    """
    Add a 'goalie_stops' feature to a tidy DataFrame of NHL shots and goals.

    Parameters:
    df (pd.DataFrame): A DataFrame with one row per shot on goal or goal, in chronological order within each game.
                       Columns include: game_id, event_id, event_type, goalie.
                       event_type is either "shot-on-goal" or "goal".
                       goalie is the goalie's full name, missing when the net is empty.

    Returns:
    pd.DataFrame: A copy of df with an additional column 'goalie_stops' indicating the number of shots stopped
                  by the goalie so far in the game (including the current event).
                  - For "shot-on-goal": increment the goalie's stop count by 1.
                  - For "goal": start the count at 0 (since the shot was not stopped).
                  - If goalie is missing (empty net), goalie_stops is missing (NaN).
    """
    df_copy = df.copy()

    # Group by game_id and goalie to track stops per goalie per game
    # We'll use a dictionary to store the current stop count for each (game_id, goalie) pair
    goalie_stops_dict = {}

    for _, row in df_copy.iterrows():
        game_id = row["game_id"]
        goalie = row["goalie"]
        event_type = row["event_type"]

        # If goalie is missing (empty net), skip setting goalie_stops
        if pd.isna(goalie):
            df_copy.at[row.name, "goalie_stops"] = pd.NA
            continue

        # Create a key for the game and goalie
        key = (game_id, goalie)

        # Initialize stop count for this goalie in this game if not already present
        if key not in goalie_stops_dict:
            goalie_stops_dict[key] = 0

        # If it's a goal, reset the stop count to 0
        if event_type == "goal":
            goalie_stops_dict[key] = 0
        # If it's a shot-on-goal, increment the stop count
        elif event_type == "shot-on-goal":
            goalie_stops_dict[key] += 1

        # Assign the current stop count to the row
        df_copy.at[row.name, "goalie_stops"] = goalie_stops_dict[key]

    return df_copy
