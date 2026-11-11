"""
Raw data extraction (code gathered from milestone 1 notebook and functions for ease of use).

Source: Milestone 1, William Perktold, ift6758/notebooks/raw_data_acquisition.ipynb
"""
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import requests

from ift6758.features.tidy import json2tidy

# AI solution to assure the following functions work from any directory 
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Code by William Perktold
# uses some code from Demo 2
class NHLyear:
    def __init__(self, year, filepath):
        """
        Creates a NHLyear object with year and filepath
        :param year: 4 digit int or string corresponding to the year the season starts (ex: 2023 for the 2023-2024 season)
        :param filepath: (string) the name of the folder where the raw data will be/is saved
        """
        self.year = str(year)
        self.filepath = filepath

    def download(self, game_id, dir_path, verbose=False):
        """
        Includes code copied from demo 1
        Downloads the raw data from NHL for the given game_id and saves it to the files
        :param game_id: int
        :param dir_path: (string) the name of the folder where the raw data will be/is saved
        :param verbose: (bool) True prints the file path and other info, False doesn't print anything
        :return: http status code AND json content (play by play)
        """

        raw_dir = Path(dir_path) / self.year
        raw_dir.mkdir(parents=True, exist_ok=True)  # create the folders if necessary
        raw_path = raw_dir / f"{game_id}.json"  # path of game folder

        if raw_path.exists():
            # game already downloaded
            if verbose:
                print("Already downloaded — reusing", raw_path.name)
            return "200", json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            # game not yet downloaded, retrieve from NHL API
            url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
            response = requests.get(url, timeout=30)

            if verbose:
                print("HTTP status:", response.status_code)
                print("Content type:", response.headers.get("Content-Type"))

            if str(response.status_code) != "200":
                # A status_code other than 200 means it was unsuccessful,
                # return status_code immediately
                return response.status_code, ""

            downloaded_game = response.json()

            # Check we received the requested game's event data before saving it.
            check = (str(downloaded_game["id"]) == str(game_id)) & isinstance(downloaded_game["plays"], list)
            if check:
                # save the game to the file
                raw_path.write_text(json.dumps(downloaded_game, indent=2), encoding="utf-8")
                if verbose:
                    print("Saved", game_id)
            else:
                print("Wrong game found?")
                print(str(downloaded_game["id"]) + "  " + game_id)

        return response.status_code, downloaded_game

    def download_all(self):
        """
        Downloads all matches from that year (regular season and playoffs). This will usually take several minutes
        """

        # loop over regular season games:
        cont = True
        i = 1
        while cont:
            game_id = self.year + "02" + f"{i:04}"  # regular season game_id is [year]02[number]
            status_code, loaded_game  = self.download(game_id, self.filepath, verbose= i % 20 == 0)
            status_code = str(status_code)
            if str(status_code) != "200":
                cont = False
                if str(status_code) != "404":
                    print(f"ERROR: {status_code}")
            i += 1
        print(f"Downloaded {i} regular season games.")
        # playoff games:
        # loop on playoff rounds :
        n = 0  # number of playoff games downloaded
        for j in range(1, 5):
            # loop over the matchups :
            for k in range(1, 2 **(4-j)+1 ):
                l = 1
                # loop over the individual games:
                while True:
                    game_id = self.year + "03" + f"{j:02}" + f"{k:01}" + f"{l:01}"  # verify?
                    l += 1
                    status_code, loaded_game = self.download(game_id, self.filepath,
                                                    verbose = (l==2))
                    if str(status_code) != "200":
                        if str(status_code) != "404":
                            print(f"ERROR Status Code: {status_code}")
                        break
                    if l>10:
                        print(f"Error {l}")
                    n += 1

        print(f"Downloaded {n} playoff games.")

    def game(self, game_id, dir_path = DATA_DIR):
        """
        Function to download and return only one game. Doesn't save to the game_list. Works even without download_all() being called first.
        :param game_id: (int or string) game id of the game to download
        :param dir_path: (str) the name of the folder where the raw data will be/is saved
        :return: the game json content OR http status code (if game isn't found)
        """
        if str(game_id)[:4] != self.year:
            return f"Game {game_id} not in {self.year}"
        raw_dir = Path(dir_path)
        raw_dir.mkdir(parents=True, exist_ok=True)

        raw_path = raw_dir / self.year / f"{game_id}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        status_code, loaded_game = self.download(game_id, dir_path,  verbose=True)

        if str(status_code) == "200":
            return loaded_game
        return status_code

def multi_year(start_year, end_year):
    """
    Download several consecutive years.
    :param start_year: the start year of the first season to download
    :param end_year: the start year of the last season to download
    """
    #inclusive
    year = start_year
    while year <= end_year:
        season_obj = NHLyear(year, DATA_DIR)
        season_obj.download_all()
        year += 1




