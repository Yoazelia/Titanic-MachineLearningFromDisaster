import pandas as pd

df = pd.read_csv("../../data/raw/train.csv")
df_test = pd.read_csv("../../data/raw/test.csv")


def preprocess_dataframe(df):
    # Fill missing values in the "Age" column with the median
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Fill missing values in the "Embarked" column with the mode
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Create a new column "FamilySize" by adding "Parch" and "SibSp"
    df["FamilySize"] = df["Parch"] + df["SibSp"]

    # Drop specified columns
    drop_columns = ["Cabin", "Ticket", "PassengerId", "Name", "SibSp", "Parch"]
    df.drop(columns=drop_columns, inplace=True)

    return df


df = preprocess_dataframe(df)

df_test = preprocess_dataframe(df_test)

df.info()
df_test.info()

df.to_pickle("../../data/interim/preprocessed.pkl")
df_test.to_pickle("../../data/interim/preprocessed_test.pkl")
