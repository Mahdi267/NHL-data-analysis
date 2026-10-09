import os
import json
from datetime import datetime


def create_log(log_dir="logs"):
    """Fonction qui génère un nouveau fichier pour enregistrer les logs et retourne le chemin vers ce fichier

    :return: Path of log file
    """
    os.makedirs(log_dir, exist_ok=True)
    nom = (
        "chat_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "jsonl"
    )  # le jsonl permet d'ajouter une nouvelle ligen sans refaire tou le json
    path = os.path.join(log_dir, nom)
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
        (result["n_token_out"] / (result["duration_s"]))
        if result["duration_s"] > 0
        else None
    )
    extra = extra or {}
    record = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "tokens_per_s": round(tokens_per_s, 2),
    }
    record = {**record, **result, **extra}  # ** permet d'unpack les dictonnaires
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )  # ensure_ascii permet les accents éèà
