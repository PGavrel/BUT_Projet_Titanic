# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path="Donnees/Donnees.csv"):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    df = df.drop(columns=["Name", "Ticket", "Cabin"])
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    df = df.fillna(df.mean())
    return df


def split_data(df, target_column="Survived"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
