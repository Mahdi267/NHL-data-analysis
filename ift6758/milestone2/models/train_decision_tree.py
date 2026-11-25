# La métodologie du Gridsearch, du Pipeline et du Fientuning est fortwmwnt inspiré de l'entrainement des modèles du projet NHL_Game_Prediction
# Source 1: NHL_Game_Prediction/Modeling Final.ipynb de gschwaeb lien Github: https://github.com/gschwaeb/NHL_Game_Prediction/blob/master/Modeling%20Final.ipynb

import time

import joblib
import numpy as np
import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate

from ift6758.milestone2.models.data import load_features, split_data_train, TRAIN_SEASONS
from ift6758.milestone2.models.train_baselines import (
    SEED, VALIDATION_SIZE, WANDB_ENTITY, WANDB_PROJECT, MODEL_DIR, FIGURE_DIR,
)
from ift6758.milestone2.models.plot import all_plot

# CONSTANTES
NUMERIC_FEATURES = [
    "game_seconds", "period", "x_coord", "y_coord", "distance", "angle",
    "last_x_coord", "last_y_coord", "time_from_last", "distance_from_last",
    "rebound", "change_angle", "speed", "empty_net",
]
# GridSearch values
PARAM_GRID = {
    "tree__max_depth": [4, 6, 8, 10, 12],
    "tree__min_samples_leaf": [50, 200, 500, 1000],
    "tree__criterion": ["gini", "entropy"],
}

CATEGORICAL_FEATURES = ["shot_type", "last_event_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Wandb
WANDB_GROUP = "decision_tree"

def load_data():
    """Load the task 4 dataset. Based on train_XGBoost"""
    df = load_features()
    return split_data_train(df, FEATURES, seed=SEED, validation_size=VALIDATION_SIZE)

# ici **tree_params permet de passer tous les hyperparamètres de l'arbre
def build_pipeline(**tree_params) -> Pipeline:
    """
    One-hot encode the categorical features, keep the numeric ones as they are.
    When done, fits a decision tree.
    param:
        **tree_params : Hyperparameters of DecisionTreeClassifier (ex: max_depth=6)

    return:
        Decision tree pipeline

    """
    # Traitement des données différent selon les colonnes et préalables à l'entrainement (inspirée par Source 1)
    preprocess = ColumnTransformer(
        transformers=[
            # Transforme données catétorique sans importance d'ordre en one-hot (shot_type,last_event_type)
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore" # Ignore pour les catégories pas vu https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    # Arbre
    tree = DecisionTreeClassifier(random_state=SEED, **tree_params)

    return Pipeline([("preprocess", preprocess), ("tree", tree)])

# GridSearchcv
def tune_tree(X_train, y_train, param_grid=PARAM_GRID, n_splits=5) -> GridSearchCV:
    """
    Search the best tree hyperparameters with stratified k-fold cross-validation on the training set.

    param:
        X_train (pd.DataFrame) : Training features
        y_train (pd.Series) : Training target (is_goal)
        param_grid (dict) : Grid of hyperparameters
        n_splits (int) : Number of folds

    return:
        GridSearchCV : Fitted search
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        build_pipeline(),
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search

def tree_observation(model: Pipeline, max_depth: int = 3):
    """
    Show the first levels of the fitted tree as text and as a figure.

    param: 
        model (Pipeline) : Fitted pipeline
        max_depth (int) : Number of levels to display
        
    return: 
        tuple[str, plt.Figure] : Text of the yes/no questions, figure of the tree
    """
    tree = model.named_steps["tree"]
    names = model.named_steps["preprocess"].get_feature_names_out()

    # Afficher question oui/non de l'arbre en texte
    text = export_text(tree, feature_names=list(names), max_depth=max_depth, show_weights=True, decimals=2)

    # Afficher question oui/non de l'arbre en graphique
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        tree,
        max_depth=max_depth,
        feature_names=names,
        class_names=["no goal", "goal"],
        filled=True,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    return text, fig

def evaluate(model, X_val, y_val) -> dict:
    """
    Validation scores of a fitted pipeline.

    return:
        dict : AUC (ranking of shots, higher is better) and log loss (quality of probabilities, lower is better)
    """
    proba = model.predict_proba(X_val)[:, 1]
    return {
        "val_auc": roc_auc_score(y_val, proba),
        "val_log_loss": log_loss(y_val, proba),
    }


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    print(X_train.shape, X_val.shape, y_train.mean())
    model = build_pipeline().fit(X_train, y_train) # modèle decision tree de base
    print(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    search = tune_tree(X_train, y_train)
    print(search.best_params_, -search.best_score_)
    results = pd.DataFrame(search.cv_results_)
    model = search.best_estimator_
    text, fig = tree_observation(model=model, max_depth=3)
    print(text)
    fig.savefig(FIGURE_DIR / "DT_tree_top.png", dpi=150, bbox_inches="tight")