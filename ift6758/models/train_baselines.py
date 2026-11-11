import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ift6758.models.data import load_features, split_data_train, REPO_ROOT
from ift6758.models.plot import all_plot

FIGURE_DIR = REPO_ROOT / "figures" / "milestone2"

def qs3_1(df):
    # Use only the distance feature
    X_train, X_val, y_train, y_val = split_data_train(df, ['distance'])

    # Train the logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Make predictions on the validation set
    prediction = clf.predict(X_val)

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

    X_train, X_val, y_train, y_val = split_data_train(df, ["distance", "angle"])

    models = {}
    probas = {}

    # Train one model for each feature set
    for name, features in feature_sets.items():
        clf = LogisticRegression()
        clf.fit(X_train[features], y_train)

        models[name] = clf
        probas[name] = clf.predict_proba(X_val[features])[:, -1]

    # Generate random probabilities for comparison
    rng = np.random.default_rng(42)
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