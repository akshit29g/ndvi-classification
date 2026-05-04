# Imports
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import accuracy_score
from lightgbm.callback import early_stopping

# Load Data


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    test_ids = test["ID"]

    train.drop(columns=["Unnamed: 0", "ID"], inplace=True, errors="ignore")
    test.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    return train, test, test_ids


def main():
    train, test, test_ids = load_data()

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    train = feature_engineering(train)
    test = feature_engineering(test)
    X = train.drop(columns=["class"])
    y = train["class"]
    X_test = test.drop(columns=["ID"], errors="ignore")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = train_model(X, y_encoded)
    best_params = tune_model(X_train, y_train, X_val, y_val)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
    )


X, X_test = preprocess(X, X_test)

if __name__ == "__main__":
    main()


# Feature Engineering
def feature_engineering(df):
    df = df.copy()

    # NDVI
    if "NIR" in df.columns and "Red" in df.columns:
        df["NDVI"] = (df["NIR"] - df["Red"]) / (df["NIR"] + df["Red"] + 1e-5)

    # Ratio
    if "Blue" in df.columns and "Green" in df.columns:
        df["B_G_ratio"] = df["Blue"] / (df["Green"] + 1e-5)

    return df


def preprocess(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test


def train_model(X, y):
    model = LGBMClassifier(n_estimators=100)
    model.fit(X, y)
    return model


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

    return study.best_params


final_model = LGBMClassifier(n_estimators=1500, **best_params)
final_model.fit(X, y_encoded)

y_pred = final_model.predict(X_test)
y_labels = le.inverse_transform(y_pred)

submission = pd.DataFrame({
    "ID": test_ids,
    "class": y_labels
})

submission.to_csv("submission.csv", index=False)
