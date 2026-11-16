import wandb
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from ift6758.milestone2.models.data import load_features, split_data_train, REPO_ROOT, TRAIN_SEASONS
from ift6758.milestone2.models.train_baselines import SEED,VALIDATION_SIZE, WANDB_ENTITY, WANDB_PROJECT, MODEL_DIR, FIGURE_DIR
from ift6758.milestone2.models.plot import all_plot

def question1():
    #same dataset using only distance and angle features
    # Don't need to log this model to the model registry
    features_df = load_features()
    X_train, X_val, y_train, y_val = split_data_train(features_df, ['distance','angle'], seed=SEED, validation_size=VALIDATION_SIZE)

    name = "XGBoost distance + angle"
    config = {
        "model": "XGBoost",
        "training_size": len(X_train),
        "validation_ratio": VALIDATION_SIZE,
        "features": ["distance", "angle"],
        "seasons": TRAIN_SEASONS,
        "game_type": "02",  # regular season
        "seed": SEED,
        "dataset": "build_dataset_q4",
    }
    # Set up tags for specific run
    tags = ["Question5.1", "XGBoost", "distance&angle_combination"]
    # run wandb for that run
    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, tags=tags, config=config
    )

    # Train
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)

    # Make predictions on the validation set
    proba = {}
    proba['XGBoost'] = clf.predict_proba(X_val)[:, -1]

    # Save the model on disk, then upload it as a wandb artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name.replace(' ', '_').replace('+', 'plus')}.joblib"
    joblib.dump(clf, model_path)
    run.log_artifact(model_path, name=model_path.stem, type="model")
    # End run and ready for next run
    run.finish()
    
    # Figures
    all_plot(y_val,proba, FIGURE_DIR,"XGBoost_q1")

    return

def question2():
    #all features created in task 4
    # and hyperparameters tuning
    return

if __name__ == "__main__":
    question1()