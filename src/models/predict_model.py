import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

df = pd.read_pickle("../../data/processed/engineered_test.pkl")
model = pickle.load(open("../../models/titanic_rf_model", "rb"))

test = pd.read_csv("../../data/raw/test.csv")

predictions = pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(df)}
)

predictions.to_csv("../../data/processed/predictions.csv", index=False)
