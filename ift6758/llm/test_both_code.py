from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from pathlib import Path

DOCS_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "raw"
)
"""Fonctions pour faire les trois tests demandés par l'énoncé """

def test_check_download(game_id,filename="2022030411.json",docs_dir=DOCS_DIR):
    """On a un dossier temporaire pour agir comme cache vide et forcer le téléchargment
    On a un Mock qui simule une request 'response' et dois agir comme resposne , donc retourner raise_for_status et une méthode json"""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    with TemporaryDirectory() as temp_dir:
        folder =  Path(Path(docs_dir) / filename)
        response = Mock()
        response.json.return_value = game
