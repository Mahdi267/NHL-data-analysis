import requests
import json
from pathlib import Path
from ift6758.llm.generated import fetch_game_llm


def fetch_season_llm(season, game_type, cache_dir):
    """
    Fetch, cache, and load all NHL play-by-play games of a given season and game type.

    Args:
        season (int): The season year (e.g., 2016 for 2016-2017 season).
        game_type (int): Game type, 2 for regular season, 3 for playoffs.
        cache_dir (str): Directory path where cached game data will be stored.

    Returns:
        list[dict]: A sorted list of game dictionaries, each containing 'id' (int) and 'plays' (list).

    Raises:
        requests.exceptions.RequestException: If the API request fails (except for 404).
        ValueError: If cached data is invalid or inconsistent.
    """
    game_ids = []
    base_season_str = f"{season:04d}"
    game_type_str = f"{game_type:02d}"

    # Generate game IDs based on format: YYYYMMXXXX
    # Regular season: 0001, 0002, ... until first missing game
    # Playoffs: 0 + round + matchup + game (one digit each), skip to next matchup on missing game

    if game_type == 2:  # Regular season
        game_num = 1
        while True:
            game_id_str = f"{base_season_str}{game_type_str}{game_num:04d}"
            game_id = int(game_id_str)
            try:
                game_data = fetch_game_llm(game_id, cache_dir)
                game_ids.append(game_data)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    break
                else:
                    raise
            except Exception as e:
                raise
            game_num += 1

    elif game_type == 3:  # Playoffs
        # Playoffs have a structure: 0 + round + matchup + game
        # Round: 1-4 (1 digit), matchup: 1-8 (1 digit), game: 1-4 (1 digit)
        # So game_id format: 2016030123 = round 1, matchup 2, game 3
        round_num = 1
        while round_num <= 4:
            matchup_num = 1
            while matchup_num <= 8:
                game_num = 1
                while game_num <= 4:
                    game_id_str = f"{base_season_str}{game_type_str}{round_num:01d}{matchup_num:01d}{game_num:01d}"
                    game_id = int(game_id_str)
                    try:
                        game_data = fetch_game_llm(game_id, cache_dir)
                        game_ids.append(game_data)
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 404:
                            game_num += 1
                            continue
                        else:
                            raise
                    except Exception as e:
                        raise
                    game_num += 1
                matchup_num += 1
            round_num += 1

    # Sort by game_id
    return sorted(game_ids, key=lambda x: x["id"])
