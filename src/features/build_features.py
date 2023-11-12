import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_pickle("../../data/interim/preprocessed.pkl")
df_test = pd.read_pickle("../../data/interim/preprocessed_test.pkl")


def feature_engineering(df):
    """
    Performs feature engineering on the Titanic dataset.

    Args:
        df (pd.DataFrame): Titanic dataset.

    Returns:
        pd.DataFrame: Processed dataset.
    """
    # Create AgeGroup
    bins = [0, 2, 12, 110]
    labels = ["Infant", "Child", "Adult"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    AgeGroup = pd.get_dummies(df["AgeGroup"], prefix="AgeGroup")
    df = df.join(AgeGroup).drop(["Age", "AgeGroup"], axis=1)

    # Create FamilyGroup
    df["FamilyGroup"] = pd.cut(
        df["FamilySize"],
        bins=[-1, 0, 3, float("inf")],
        labels=["Alone", "Small Family", "Large Family"],
        right=False,
    )
    Family = pd.get_dummies(df["FamilyGroup"], prefix="Family")
    df = df.join(Family).drop(["FamilyGroup", "FamilySize"], axis=1)

    # Create HighFare
    df["HighFare"] = (df["Fare"] >= 100).astype(int)
    df.drop("Fare", axis=1, inplace=True)

    # One-hot encode Embarked
    embarked = pd.get_dummies(df["Embarked"], prefix="Embarked")
    df = df.join(embarked).drop("Embarked", axis=1)

    # Binary encode Sex
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    return df


df = feature_engineering(df)
df_test = feature_engineering(df_test)

df.to_pickle("../../data/processed/engineered.pkl")
df_test.to_pickle("../../data/processed/engineered_test.pkl")
