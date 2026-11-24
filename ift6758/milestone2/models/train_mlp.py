from ift6758.milestone2.models.data import load_features, split_data_train
from ift6758.milestone2.models.train_baselines import SEED, VALIDATION_SIZE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, roc_auc_score

NUMERIC_FEATURES = [
    "game_seconds", "period", "x_coord", "y_coord", "distance", "angle",
    "last_x_coord", "last_y_coord", "time_from_last", "distance_from_last",
    "rebound", "change_angle", "speed",
]
CATEGORICAL_FEATURES = ["shot_type", "last_event_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_preprocess():
    # Transformer to rescale all numerical value to the 
    # same order of magnitude and encode categorical features
    # using one-hot encoding
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )

if __name__ == "__main__" :
    df = load_features()

    X_train, X_val, y_train, y_val = split_data_train(
        df, FEATURES, validation_size=VALIDATION_SIZE, seed=SEED
    )

    print(df.shape)
    # En retirant les NaN, on est passé de 520 564 à 511 524 tirs.
    # On a donc retiré environ 9 000 lignes. La proportion de buts
    # a aussi baissé, de 0,0953 à 0,0932.
    print(X_train.shape, X_val.shape)
    print(y_train.mean(), y_val.mean())

    # Les NaN retirés représentent une partie importante des buts, 
    # donc des tirs plus dangereux que la moyenne. À noter comme 
    # limite et piste : remplacer les NaN plutôt que les supprimer.

    preprocess = build_preprocess()
    X_train_prep = preprocess.fit_transform(X_train)
    X_val_prep = preprocess.transform(X_val)

    print(X_train_prep.shape, X_val_prep.shape)
    # Check whether "distance" mean is close to 0 and standard deviation to 1
    print(X_train_prep[:, 4].mean(), X_train_prep[:, 4].std())

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        early_stopping=True,
        random_state=SEED
    )
    mlp.fit(X_train_prep, y_train)

    # Probability that the shot is a goal (1)
    proba_val = mlp.predict_proba(X_val_prep)[:, 1]

    print(mlp.n_iter_)
    print(roc_auc_score(y_val, proba_val))
    print(log_loss(y_val, proba_val))