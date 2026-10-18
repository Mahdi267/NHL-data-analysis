"""
Logging of the Qwen calls: one JSONL file per session, one line per call.
"""

from pathlib import Path
import json
from datetime import datetime

LOGS_DIR = (
    Path(__file__).resolve().parents[2] / "logs"
)

def create_log(log_dir=LOGS_DIR):
    """Create the log folder if needed and return the path of a new timestamped log file.

    :param log_dir: Log folder
    :type log_dir: Path or str
    :return: Path of the log file
    :rtype: Path
    """
    nom = (
        "chat_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jsonl"
    )  # JSONL lets us append a new line without rewriting the whole JSON
    path = Path(Path(log_dir) / nom)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def log_call(log_path, messages, result, extra=None):
    """Append one call (messages, result, tokens/s and extra info) to the JSONL log.

    :param log_path: Log file from create_log
    :type log_path: Path or str
    :param messages: Messages sent to the model
    :type messages: list[dict]
    :param result: Output of generate()
    :type result: dict
    :param extra: Extra info to log
    :type extra: dict or None
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
    record = {**record, **result, **extra}  # ** unpacks the dictionaries
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )  # ensure_ascii=False keeps accents (éèà) readable
