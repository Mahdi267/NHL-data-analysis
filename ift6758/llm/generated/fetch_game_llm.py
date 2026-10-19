import requests
import json
from pathlib import Path


def fetch_game_llm(game_id, cache_dir):
    """
    Fetch play-by-play data for a given game ID from the NHL API with caching.

    Args:
        game_id (int): The ID of the game to fetch data for.
        cache_dir (str): Directory path where cached game data will be stored.

    Returns:
        dict: The full game dictionary containing 'id' (int) and 'plays' (list).

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        ValueError: If cached data exists but has a different game_id.
        json.JSONDecodeError: If cached data is invalid JSON.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    cache_path = Path(cache_dir) / f"{game_id}.json"

    # Create cache directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if cached data exists
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        # Validate cached data
        if not isinstance(cached_data, dict):
            raise ValueError("Cached data is not a valid dictionary.")
        if "id" not in cached_data or not isinstance(cached_data["id"], int):
            raise ValueError("Cached data missing or invalid 'id' field.")
        if "plays" not in cached_data or not isinstance(cached_data["plays"], list):
            raise ValueError("Cached data missing or invalid 'plays' field.")

        # Validate game_id consistency
        if cached_data["id"] != game_id:
            raise ValueError(
                f"Cached game_id ({cached_data['id']}) does not match requested game_id ({game_id})"
            )

        return cached_data

    # Fetch data from API
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Validate response structure
        if not isinstance(data, dict):
            raise ValueError("API response is not a valid dictionary.")
        if "id" not in data or not isinstance(data["id"], int):
            raise ValueError("API response missing or invalid 'id' field.")
        if "plays" not in data or not isinstance(data["plays"], list):
            raise ValueError("API response missing or invalid 'plays' field.")

        # Save to cache
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        return data

    except requests.exceptions.RequestException as e:
        raise e
