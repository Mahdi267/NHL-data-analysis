import wandb
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from ift6758.milestone2.models.data import load_features, split_data_train, REPO_ROOT, TRAIN_SEASONS
from ift6758.milestone2.models.plot import all_plot

FIGURE_DIR = REPO_ROOT / "figures" / "milestone2"
MODEL_DIR = REPO_ROOT / "models" / "milestone2"
WANDB_ENTITY = 'IFT6758A_2026_Team5'
WANDB_PROJECT = 'IFT6758A_2026_05_milestone2'
SEED = 42
VALIDATION_SIZE = 0.2

def qs3_1(df):
    # Use only the distance feature
    X_train, X_val, y_train, y_val = split_data_train(df, ['distance'], seed=SEED, validation_size=VALIDATION_SIZE)

    name = "Distance"
    config = {
        "model": "LogisticRegression",
        "training_size": len(X_train),
        "validation_ratio": VALIDATION_SIZE,
        "features": "distance",
        "seasons": TRAIN_SEASONS,
        "game_type": "02",  # regular season
        "seed": SEED,
        "dataset": "build_dataset_q2",
    }
    # Set up tags for specific run
    tags = ["Question3.1", "LogisticRegression", "distance"]
    # run wandb for that run
    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, tags=tags, config=config
    )
    # Train logistic Regression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Make predictions on the validation set
    prediction = clf.predict(X_val)
    
    # log validation metric
    run.log({"accuracy": accuracy_score(y_val,prediction)})

    # Save the model on disk, then upload it as a wandb artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name.replace(' ', '_').replace('+', 'plus')}.joblib"
    joblib.dump(clf, model_path)
    run.log_artifact(model_path, name=model_path.stem, type="model")
    # End run and ready for next run
    run.finish()
    

    # Compare the model accuracy with the no-goal baseline
    print(f"Validation accuracy: {accuracy_score(y_val, prediction):.4f}")
    print(f"Accuracy if we always predict 'no goal' for each event: {1 - y_val.mean():.4f}")

    return clf


def qs3_2_3(df):
    # Test different combinations of features
    feature_sets = {
        "LR distance": ["distance"],
        "LR angle": ["angle"],
        "LR distance + angle": ["distance", "angle"],
    }

    X_train, X_val, y_train, y_val = split_data_train(df, ["distance", "angle"], seed=SEED, validation_size=VALIDATION_SIZE)

    models = {}
    probas = {}

    # Train one model for each feature set
    for name, features in feature_sets.items():
        # Set up configuration information for wandb
        config = {
            "model": "LogisticRegression",
            "training_size": len(X_train),
            "validation_ratio": len(X_val) / len(df),
            "features": features,
            "seasons": TRAIN_SEASONS,
            "game_type": "02",  # regular season
            "seed": SEED,
            "dataset": "build_dataset_q2",
        }
        # Set up tags for specific run
        tags = ["Question3.2",
                "LogisticRegression",
                "distance&angle_combination"
                ] 
        # run wandb for that run
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, tags=tags, config=config)
        clf = LogisticRegression()
        clf.fit(X_train[features], y_train)

        models[name] = clf
        probas[name] = clf.predict_proba(X_val[features])[:, -1]
        # log validation metric
        run.log({"val_auc": roc_auc_score(y_val, probas[name])})

        # Save the model on disk, then upload it as a wandb artifact
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"{name.replace(' ', '_').replace('+', 'plus')}.joblib"
        joblib.dump(clf, model_path)
        run.log_artifact(model_path, name=model_path.stem, type="model")
        # End run and ready for next run
        run.finish()
    # Generate random probabilities for comparison
    rng = np.random.default_rng(SEED)
    probas["Random baseline"] = rng.uniform(0, 1, len(y_val))

    print(len(y_val), {name: len(p) for name, p in probas.items()})

    # Plot the results for all models
    all_plot(y_val, probas, FIGURE_DIR, "q3_baselines")
    print(f"Figures saved in {FIGURE_DIR}")

    return models


if __name__ == "__main__":
    # Load the features and run the questions
    features_df = load_features()
    qs3_1(features_df)
    qs3_2_3(features_df)