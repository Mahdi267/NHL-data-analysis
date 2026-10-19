import pandas as pd
import json
from typing import Union

def json2tidy_llm(raw_data: Union[str, dict]) -> pd.DataFrame:
    """
    Convert NHL game events (shots on goal and goals) into a tidy pandas DataFrame.
    
    Parameters:
    raw_data (str or dict): Either a path to a JSON file (read as UTF-8) or a dict containing the data.
    
    Returns:
    pd.DataFrame: A DataFrame with columns: game_id, event_id, season, game_type, period, time,
                  team_id, event_type, x_coord, y_coord, shooter, goalie, shot_type, empty_net, strength.
    - event_type is "shot-on-goal" or "goal".
    - game_type is the numeric game type.
    - shooter and goalie are full names (e.g., "First Last").
    - empty_net is True or False.
    - strength is "even", "shorthanded", or "power play" for goals; missing for shots.
    - Missing values are preserved as NaN.
    """
    # Load data
    if isinstance(raw_data, str):
        with open(raw_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = raw_data
    
    # Extract events (assume data structure has 'events' or similar)
    events = data.get('events', []) if isinstance(data, dict) else []
    
    # Initialize list for results
    rows = []
    
    for event in events:
        event_id = event.get('event_id')
        game_id = event.get('game_id')
        season = event.get('season')
        game_type = event.get('game_type')
        period = event.get('period')
        time = event.get('time')
        team_id = event.get('team_id')
        event_type = event.get('event_type')
        
        # Only keep shots on goal and goals
        if event_type not in ['shot_on_goal', 'goal']:
            continue
            
        # Extract coordinates
        x_coord = event.get('x')
        y_coord = event.get('y')
        
        # Extract shooter and goalie
        shooter = event.get('shooter') or event.get('player_name')
        goalie = event.get('goalie') or event.get('goalie_name')
        
        # Format names to "First Last"
        shooter = shooter.split(' ')[-1] if shooter and isinstance(shooter, str) else ""
        goalie = goalie.split(' ')[-1] if goalie and isinstance(goalie, str) else ""
        
        # Extract shot type
        shot_type = event.get('shot_type')
        
        # Empty net flag
        empty_net = event.get('empty_net', False)
        
        # Strength for goals only
        strength = None
        if event_type == 'goal':
            strength = event.get('strength')
        
        # Add row
        rows.append({
            'game_id': game_id,
            'event_id': event_id,
            'season': season,
            'game_type': game_type,
            'period': period,
            'time': time,
            'team_id': team_id,
            'event_type': event_type,
            'x_coord': x_coord,
            'y_coord': y_coord,
            'shooter': shooter,
            'goalie': goalie,
            'shot_type': shot_type,
            'empty_net': empty_net,
            'strength': strength
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure correct data types and missing values
    df['event_type'] = df['event_type'].str.upper()
    df['event_type'] = df['event_type'].replace({'SHOT_ON_GOAL': 'shot-on-goal', 'GOAL': 'goal'})
    
    # Ensure strength is only for goals and missing for shots
    df['strength'] = df['strength'].fillna('')
    df['strength'] = df['strength'].replace({'even': 'even', 'shorthanded': 'shorthanded', 'power play': 'power play'})
    
    # Convert to proper types
    df['game_id'] = pd.to_numeric(df['game_id'], errors='coerce')
    df['event_id'] = pd.to_numeric(df['event_id'], errors='coerce')
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['game_type'] = pd.to_numeric(df['game_type'], errors='coerce')
    df['period'] = pd.to_numeric(df['period'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
    
    # Ensure event_type is string
    df['event_type'] = df['event_type'].astype(str)
    
    return df
