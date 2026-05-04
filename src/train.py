# Imports
import pandas as pd

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
