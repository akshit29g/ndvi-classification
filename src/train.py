import pandas as pd
import numpy as np
import optuna
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping


# =========================
# Load Data
# =========================
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    test_ids = test["ID"]

    train.drop(columns=["Unnamed: 0", "ID"], inplace=True, errors="ignore")
    test.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    return train, test, test_ids


# =========================
# Feature Engineering
# =========================
def feature_engineering(df):
    df = df.copy()

    # NDVI (if NIR & Red exist)
    if "NIR" in df.columns and "Red" in df.columns:
        df["NDVI"] = (df["NIR"] - df["Red"]) / (df["NIR"] + df["Red"] + 1e-5)

    # Ratio feature
    if "Blue" in df.columns and "Green" in df.columns:
        df["B_G_ratio"] = df["Blue"] / (df["Green"] + 1e-5)

    # Time-series aggregations (IMPORTANT UPGRADE)
    ndvi_cols = [col for col in df.columns if "_N" in col]

    if len(ndvi_cols) > 0:
        df["ndvi_mean"] = df[ndvi_cols].mean(axis=1)
        df["ndvi_std"] = df[ndvi_cols].std(axis=1)
        df["ndvi_max"] = df[ndvi_cols].max(axis=1)
        df["ndvi_min"] = df[ndvi_cols].min(axis=1)

    return df


# =========================
# Preprocessing
# =========================
def preprocess(X_train, X_test):
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, X_test


# =========================
# Optuna Optimization
# =========================
def tune_model(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "n_estimators": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
        }

        model = LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50)],
        )

        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best Accuracy:", study.best_value)
    print("Best Params:", study.best_params)

    return study.best_params


# =========================
# Train Final Model
# =========================
def train_model(X, y, best_params):
    model = LGBMClassifier(
        n_estimators=1500,
        random_state=42,
        **best_params
    )

    model.fit(X, y)
    return model


# =========================
# Main Pipeline
# =========================
def main():
    train, test, test_ids = load_data()

    # Feature Engineering
    train = feature_engineering(train)
    test = feature_engineering(test)

    # Split features and target
    X = train.drop(columns=["class"])
    y = train["class"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_test = test.drop(columns=["ID"], errors="ignore")

    # Preprocessing
    X, X_test = preprocess(X, X_test)

    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
    )

    # Hyperparameter tuning
    best_params = tune_model(X_train, y_train, X_val, y_val)

    # Train final model
    final_model = train_model(X, y_encoded, best_params)

    # Predictions
    y_pred = final_model.predict(X_test)
    y_labels = le.inverse_transform(y_pred)

    # Save submission
    submission = pd.DataFrame({
        "ID": test_ids,
        "class": y_labels
    })

    submission.to_csv("submission.csv", index=False)
    print("Submission file saved!")


if __name__ == "__main__":
    main()
