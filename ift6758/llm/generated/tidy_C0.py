import pandas as pd
import json
import numpy as np

def json2tidy_llm(raw_data):
    """
    Convert NHL game events (shots on goal and goals) into a tidy pandas DataFrame.
    
    Parameters:
    raw_data (str or dict): Either a path to a JSON file (read as UTF-8) or a loaded JSON dictionary.
    
    Returns:
    pandas.DataFrame: Tidy DataFrame with one row per shot or goal event, including:
        - game_id, event_id, season, game_type, period, time
        - team_id, event_type (either "shot-on-goal" or "goal")
        - x_coord, y_coord, shooter, goalie, shot_type, empty_net
        - strength (only for goals: "even", "shorthanded", "power play"; missing for shots)
    """
    # Load data
    if isinstance(raw_data, str):
        with open(raw_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = raw_data
    
    # Initialize list to store events
    events = []
    
    # Process each game
    for game in data.get('games', []):
        game_id = game.get('game_id')
        season = game.get('season')
        game_type = game.get('game_type')
        
        for event in game.get('events', []):
            event_id = event.get('event_id')
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
            shooter = event.get('shooter')
            goalie = event.get('goalie')
            
            # Extract shot type
            shot_type = event.get('shot_type')
            
            # Extract empty net
            empty_net = event.get('empty_net')
            
            # Extract strength (only for goals)
            strength = None
            if event_type == 'goal':
                strength = event.get('strength')
            
            # Build event dictionary
            event_dict = {
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
            }
            
            events.append(event_dict)
    
    # Convert to DataFrame
    df = pd.DataFrame(events)
    
    # Ensure proper data types and missing values
    df['game_id'] = df['game_id'].astype(str)
    df['event_id'] = df['event_id'].astype(str)
    df['season'] = df['season'].astype(str)
    df['game_type'] = df['game_type'].astype(int)
    df['period'] = df['period'].astype(int)
    df['time'] = df['time'].astype(str)
    df['team_id'] = df['team_id'].astype(str)
    
    # Convert boolean fields to proper booleans
    bool_columns = ['empty_net']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: True if pd.isna(x) or x is True else False)
    
    # Ensure strength is properly formatted
    df['strength'] = df['strength'].fillna('')
    df['strength'] = df['strength'].apply(lambda x: x if x in ['even', 'shorthanded', 'power play'] else '')
    
    # Filter out empty strength (only for goals)
    df = df[df['event_type'] == 'goal'].copy()
    df['strength'] = df['strength'].fillna('')
    
    # Final cleanup: ensure all required columns are present
    required_columns = ['game_id', 'event_id', 'season', 'game_type', 'period', 'time',
                        'team_id', 'event_type', 'x_coord', 'y_coord', 'shooter', 'goalie',
                        'shot_type', 'empty_net', 'strength']
    
    # Reconstruct with required columns
    df = df[required_columns]
    
    # Ensure event_type is correct
    df['event_type'] = df['event_type'].map({'shot_on_goal': 'shot-on-goal', 'goal': 'goal'})
    
    # Return DataFrame
    return df
