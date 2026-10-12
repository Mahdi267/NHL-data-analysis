from pathlib import Path
import json
from datetime import datetime

LOGS_DIR = (
    Path(__file__).resolve().parents[2] / "logs"
)

def create_log(log_dir=LOGS_DIR):
    """Fonction qui génère un nouveau fichier pour enregistrer les logs et retourne le chemin vers ce fichier
    :param log_dir: le dossier, LOGS_DIR par defaut
    :type log_dir: Path ou str
    :return: Path of log file
    :rtype: Path
    """
    nom = (
        "chat_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jsonl"
    )  # le jsonl permet d'ajouter une nouvelle ligen sans refaire tou le json
    path = Path(Path(log_dir) / nom)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def log_call(log_path, messages, result, extra=None):
    """Fonction qui enregistre les informations d'une conversation dans un fichier jsonL.

    :param log_path:
    :param messages:
    :param result:
    :param extra:
    :return: None
    """

    tokens_per_s = (
        round((result["n_token_out"] / (result["duration_s"])),2)
        if result["duration_s"] > 0
        else None
    )
    extra = extra or {}
    record = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "tokens_per_s": tokens_per_s,
    }
    record = {**record, **result, **extra}  # ** permet d'unpack les dictonnaires
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )  # ensure_ascii permet les accents éèà
