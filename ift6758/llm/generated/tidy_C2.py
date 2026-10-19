import pandas as pd
import json
from typing import Union

def json2tidy_llm(raw_data: Union[str, dict]) -> pd.DataFrame:
    """
    Convert an NHL game JSON into a tidy pandas DataFrame containing only shots on goal and goals.
    
    Input:
        raw_data: Either a string path to a JSON file (read as UTF-8) or a dictionary representing the game data.
    
    Output:
        A pandas DataFrame with the following columns:
        - game_id: the game ID (int)
        - event_id: the event ID (int)
        - season: the season (int)
        - game_type: the game type (int)
        - period: the period number (int)
        - time: time in the period (str, e.g., "01:29")
        - team_id: the team that took the shot (int)
        - event_type: "shot-on-goal" or "goal"
        - x_coord: x coordinate on ice (int)
        - y_coord: y coordinate on ice (int)
        - shooter: full name of shooter ("First Last")
        - goalie: full name of goalie ("First Last")
        - shot_type: type of shot (str)
        - empty_net: True/False if net was empty
        - strength: "even", "shorthanded", "power play" for goals; missing for shots
    
    Notes:
        - Only shots on goal and goals are included.
        - Missing values are preserved as NaN.
        - event_type is derived from typeDescKey.
        - strength is derived from situationCode (e.g., 1451 -> even strength).
        - For shots, strength is missing.
        - For goals, strength is inferred from situationCode (e.g., 1451 = even, 1452 = shorthanded, 1453 = power play).
    """
    # Load data
    if isinstance(raw_data, str):
        with open(raw_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = raw_data
    
    # Extract game metadata
    game_info = data.get("game", {})
    game_id = data.get("id")
    season = data.get("season")
    game_type = data.get("gameType")
    
    # Extract plays
    plays = data.get("plays", [])
    
    # Map situation codes to strength
    strength_map = {
        1451: "even",
        1452: "shorthanded",
        1453: "power play"
    }
    
    rows = []
    
    for play in plays:
        event = play
        event_id = event.get("eventId")
        period = event["periodDescriptor"].get("number")
        time_in_period = event.get("timeInPeriod")
        time_remaining = event.get("timeRemaining")
        situation_code = event.get("situationCode")
        team_id = event.get("eventOwnerTeamId")
        type_code = event.get("typeCode")
        type_desc_key = event.get("typeDescKey")
        
        # Only include shots on goal and goals
        if type_desc_key not in ["shot-on-goal", "goal"]:
            continue
            
        # Extract details
        details = event.get("details", {})
        x_coord = details.get("xCoord")
        y_coord = details.get("yCoord")
        zone_code = details.get("zoneCode")
        shot_type = details.get("shotType")
        shooting_player_id = details.get("shootingPlayerId")
        scoring_player_id = details.get("scoringPlayerId")
        goalie_in_net_id = details.get("goalieInNetId")
        
        # Determine event type
        event_type = "shot-on-goal" if type_desc_key == "shot-on-goal" else "goal"
        
        # Get player names from rosterSpots
        roster_spots = data.get("rosterSpots", [])
        shooter_name = None
        goalie_name = None
        
        for player in roster_spots:
            if player["playerId"] == shooting_player_id:
                shooter_name = f"{player['firstName']['default']} {player['lastName']['default']}"
            elif player["playerId"] == goalie_in_net_id:
                goalie_name = f"{player['firstName']['default']} {player['lastName']['default']}"
        
        # Determine empty net
        empty_net = details.get("emptyNet", False) if "emptyNet" in details else False
        
        # Determine strength for goals only
        strength = None
        if event_type == "goal":
            strength = strength_map.get(situation_code, None)
        
        # Append row
        rows.append({
            "game_id": game_id,
            "event_id": event_id,
            "season": season,
            "game_type": game_type,
            "period": period,
            "time": time_in_period,
            "team_id": team_id,
            "event_type": event_type,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "shooter": shooter_name,
            "goalie": goalie_name,
            "shot_type": shot_type,
            "empty_net": empty_net,
            "strength": strength
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure correct data types and handle missing values
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["game_type"] = pd.to_numeric(df["game_type"], errors="coerce")
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    
    return df
