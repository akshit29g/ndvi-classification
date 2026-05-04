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


if __name__ == "__main__":
    main()
