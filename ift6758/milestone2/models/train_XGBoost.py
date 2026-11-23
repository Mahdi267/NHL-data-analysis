import wandb
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from ift6758.milestone2.models.data import load_features, split_data_train, REPO_ROOT, TRAIN_SEASONS
from ift6758.milestone2.models.train_baselines import SEED,VALIDATION_SIZE, WANDB_ENTITY, WANDB_PROJECT, MODEL_DIR, FIGURE_DIR
from ift6758.milestone2.models.plot import roc_plot, goal_rate_plot, cumulative_goals_plot, plot_calibration


def question1(df):
    #same dataset using only distance and angle features
    X_train, X_val, y_train, y_val = split_data_train(df, ['distance','angle'], seed=SEED, validation_size=VALIDATION_SIZE)

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
    probas = {}
    probas['XGBoost baseline'] = clf.predict_proba(X_val)[:, -1]

    # Save the model on disk, then upload it as a wandb artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name.replace(' ', '_').replace('+', 'plus')}.joblib"
    joblib.dump(clf, model_path)
    run.log_artifact(model_path, name=model_path.stem, type="model")
    # End run and ready for next run
    run.finish()

    
    roc = roc_plot(y_val,probas)
    
    goal = goal_rate_plot(y_val,probas)

    cumulative = cumulative_goals_plot(y_val,probas)

    calibration = plot_calibration(y_val,probas)

    return roc, goal, cumulative, calibration

def question2(df,roc,goal,cumulative,calibration):
    # train using all features created in task 4 and hyperparameters tuning
    features = ['game_seconds','period','x_coord','y_coord','distance','angle',
                'shot_type','last_event_type','last_x_coord','last_y_coord',
                'time_from_last','distance_from_last','rebound','change_angle','speed']
    X_train, X_val, y_train, y_val = split_data_train(df, features, seed=SEED, validation_size=VALIDATION_SIZE)

    X_train['shot_type'] = X_train['shot_type'].astype('category')
    X_train['last_event_type'] = X_train['last_event_type'].astype('category')

    X_val['shot_type'] = X_val['shot_type'].astype('category')
    X_val['last_event_type'] = X_val['last_event_type'].astype('category')

    
    name = "XGBoost hyperparameter tuning"
    config = {
        "model": "XGBoost",
        "training_size": len(X_train),
        "validation_ratio": VALIDATION_SIZE,
        "features": features,
        "seasons": TRAIN_SEASONS,
        "game_type": "02",  # regular season
        "seed": SEED,
        "dataset": "build_dataset_q4",
    }
    # Set up tags for specific run
    tags = ["Question5.2", "XGBoost", "hyperparameter_tuning", "all_features"]
    # run wandb for that run
    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, tags=tags, config=config
    )

    # Train
    clf = xgb.XGBClassifier(enable_categorical = True,random_state=SEED, learning_rate = 0.1, max_depth = 7, subsample=1.0)
    clf.fit(X_train, y_train)

    # Make predictions on the validation set
    probas = {}
    probas['XGBoost hyperparameter tuning'] = clf.predict_proba(X_val)[:, -1]

    # Save the model on disk, then upload it as a wandb artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name.replace(' ', '_').replace('+', 'plus')}.joblib"
    joblib.dump(clf, model_path)
    run.log_artifact(model_path, name=model_path.stem, type="model")
    # End run and ready for next run
    run.finish()

    roc = roc_plot(y_val,probas,roc)

    goal = goal_rate_plot(y_val,probas,ax=goal)

    cumulative = cumulative_goals_plot(y_val,probas,cumulative)

    calibration = plot_calibration(y_val,probas,ax=calibration)

    return roc,goal,cumulative,calibration

def question3(df,roc,goal,cumulative,calibration):
    # Feature selection
    selection = ['game_seconds', 'period', 'distance', 'angle', 'shot_type',
       'last_event_type', 'time_from_last', 'distance_from_last', 'rebound','speed']
    X_train, X_val, y_train, y_val = split_data_train(df, selection, seed=SEED, validation_size=VALIDATION_SIZE)

    X_train['shot_type'] = X_train['shot_type'].astype('category')
    X_train['last_event_type'] = X_train['last_event_type'].astype('category')

    X_val['shot_type'] = X_val['shot_type'].astype('category')
    X_val['last_event_type'] = X_val['last_event_type'].astype('category')

    cat_cols = X_train.select_dtypes(include=['category']).columns

    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        categories = X_train[col].cat.categories

        X_val[col] = pd.Categorical(
            X_val[col],
            categories=categories
        )

        X_train[col] = X_train[col].cat.codes
        X_val[col] = X_val[col].cat.codes

    name = "XGBoost feature selection"
    config = {
        "model": "XGBoost",
        "training_size": len(X_train),
        "validation_ratio": VALIDATION_SIZE,
        "features": selection,
        "seasons": TRAIN_SEASONS,
        "game_type": "02",  # regular season
        "seed": SEED,
        "dataset": "build_dataset_q4",
    }
    # Set up tags for specific run
    tags = ["Question5.3", "XGBoost", "feature_selection"]
    # run wandb for that run
    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, tags=tags, config=config
    )

    # Train
    clf = xgb.XGBClassifier(enable_categorical = True,random_state=SEED, learning_rate = 0.1, max_depth = 7, subsample=1.0)
    clf.fit(X_train, y_train)

    # Make predictions on the validation set
    probas = {}
    probas['XGBoost feature selection'] = clf.predict_proba(X_val)[:, -1]

    # Save the model on disk, then upload it as a wandb artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name.replace(' ', '_').replace('+', 'plus')}.joblib"
    joblib.dump(clf, model_path)
    run.log_artifact(model_path, name=model_path.stem, type="model")
    # End run and ready for next run
    run.finish()

    # Figures with all results
    roc = roc_plot(y_val,probas,roc)
    roc_plt = roc.get_figure()
    roc_plt.savefig(FIGURE_DIR / "XGBoost_roc.png", dpi=150)

    goal = goal_rate_plot(y_val,probas,ax=goal)
    goal_plt = goal.get_figure()
    goal_plt.savefig(FIGURE_DIR / "XGBoost_goal_rate.png", dpi=150)

    cumulative = cumulative_goals_plot(y_val,probas,cumulative)
    cumulative_plt = cumulative.get_figure()
    cumulative_plt.savefig(FIGURE_DIR / "XGBoost_cumulative_goals.png", dpi=150)

    calibration = plot_calibration(y_val,probas,ax=calibration)
    calibration_plt = calibration.get_figure()
    calibration_plt.savefig(FIGURE_DIR / "XGBoost_calibration.png", dpi=150)

    return

if __name__ == "__main__":
    features_df = load_features()
  
    roc,goal,cumulative,calibration = question1(features_df)
    roc,goal,cumulative,calibration = question2(features_df,roc,goal,cumulative,calibration)
    question3(features_df,roc,goal,cumulative,calibration)

    