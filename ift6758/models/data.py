from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ift6758.features.tidier import build_dataset

# AI solution to assure the following functions work from any directory
REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "ift6758" / "milestone2" / "train_basic.pkl"
TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022]


def load_features(path=FEATURES_PATH, seasons=TRAIN_SEASONS) -> pd.DataFrame:
    path = Path(path)

    # Load the dataset if it already exists
    if path.exists():
        df = pd.read_pickle(path)
    else:
        # Build and save the dataset if it does not exist
        df = build_dataset(seasons)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)

    # Convert the goal events into 0 and 1
    df["is_goal"] = (df["event_type"] == "goal").astype(int)
    return df


def split_data_train(
    df_data: pd.DataFrame,
    features: list,
    validation_size: float = 0.2,
    seed: int = 42
):
    # Remove rows with missing features from features list
    df = df_data.dropna(subset=features)

    # Split the data into training and validation sets
    return train_test_split(
        df[features],
        df["is_goal"],
        test_size=validation_size,
        stratify=df["is_goal"],
        random_state=seed,
    )