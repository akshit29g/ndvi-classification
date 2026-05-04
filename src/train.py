# Imports
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
