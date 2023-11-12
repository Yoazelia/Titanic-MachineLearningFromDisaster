import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_pickle("../../data/processed/engineered.pkl")

y = df["Survived"]
df.drop("Survived", axis=1, inplace=True)


def random_forest_classifier(df, y, test_size=0.2, random_state=42):
    """
    Trains a Random Forest Classifier on the given features and target.

    Args:
        df (pd.DataFrame): Features (independent variables).
        y (pd.Series): Target variable (dependent variable).
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        RandomForestClassifier: Trained model.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state
    )

    # Initialize the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return rf_model


rf_model = random_forest_classifier(df, y)

pickle.dump(rf_model, open("../../models/titanic_rf_model", "wb"))
