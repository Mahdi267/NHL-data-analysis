import hashlib
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Fonction par Claude pour faire des hashs
def rows_fingerprint(y: pd.Series) -> str:
    """Short fingerprint of WHICH shots are in a set (index + target). Same value = same rows."""
    # fingerprint for row
    rows = pd.util.hash_pandas_object(y, index=True).values
    # fingerprint for all data
    return hashlib.sha256(rows.tobytes()).hexdigest()[:12]


def data_config(seasons,game_type,seed,val_ratio,X_train,y_train,X_val,y_val) -> dict:
    """
    Returns a dict of everything needed to know which data a model saw.

    :param seasons:
    :param game_type:
    :param seed:
    :param val_ratio:
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :return: dict
    """
    data = {
        "seasons": seasons,
        "game_type": game_type,
        "features": list(X_train.columns),
        "n_features": X_train.shape[1],
        "seed": seed,
        "validation_ratio": val_ratio,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "train_goal_rate": float(y_train.mean()),
        "val_goal_rate": float(y_val.mean()),
        "train_rows_hash": rows_fingerprint(y_train),
        "val_rows_hash": rows_fingerprint(y_val),
    }
    return data

def evaluate(model, X, y, prefix: str = 'val') -> dict:
    """
     Scores of a fitted pipeline

    :param model:
    :param X:
    :param y:
    :param prefix: 'val' or 'test'

    return:
        dict : AUC and log loss
    """
    proba = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_auc": roc_auc_score(y, proba),
        f"{prefix}_log_loss": log_loss(y, proba),
    }
