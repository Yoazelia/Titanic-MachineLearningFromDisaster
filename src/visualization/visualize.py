import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_pickle("../../data/interim/preprocessed.pkl")


def plot_titanic_data(df):
    """
    Generates and displays various plots for Titanic dataset.

    Args:
        data_df (pd.DataFrame): Titanic dataset with relevant columns.

    Returns:
        None
    """
    features_to_plot = ["Pclass", "Age", "Fare", "Sex", "Embarked", "FamilySize"]

    for feature in features_to_plot:
        plt.figure(figsize=(8, 6))
        if feature == "Age" or feature == "Fare":
            sns.histplot(df[feature], bins=20, kde=True)
        else:
            sns.barplot(x=feature, y="Survived", data=df, errorbar=None)
        plt.title(
            f"{feature} Distribution" if feature != "Sex" else "Survival Rate by Gender"
        )
        plt.xlabel(feature)
        plt.ylabel("Count" if feature in ["Age", "Fare"] else "Survival Rate")
        plt.show()


plot_titanic_data(df)
# Pclass 1
# Age 20 - 30
# Fare 0 - 100
# Gender Female
# Embarked Cherbourg
# Family size 3
