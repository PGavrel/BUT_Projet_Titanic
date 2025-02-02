import pandas as pd
import pytest

from SRC.data_preprocessing import preprocess_data, split_data


def test_preprocess_data():
    data = {
        "Age": [22, 38, 26, None],
        "Fare": [7.25, 71.2833, 8.05, 15.0],
        "Sex": ["male", "female", "female", "male"],
        "Embarked": ["S", "C", "S", "Q"],
    }
    df = pd.DataFrame(data)

    processed_df = preprocess_data(df)
    assert processed_df.isnull().sum().sum() == 0, "There should be no missing values"
    assert "Sex_male" in processed_df.columns, "Sex encoding is missing"
    assert "Embarked_Q" in processed_df.columns, "Embarked encoding is missing"
    assert "Sex" not in processed_df.columns, "Original 'Sex' column should be removed"
    assert (
        "Embarked" not in processed_df.columns
    ), "Original 'Embarked' column should be removed"


def test_split_data():
    data = {
        "Survived": [1, 0, 1, 0, 1],
        "Age": [22, 38, 26, 35, 40],
        "Fare": [7.25, 71.2833, 8.05, 15.0, 20.0],
    }
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) == 4, "Train set should have 4 samples"
    assert len(X_test) == 1, "Test set should have 1 sample"
    assert (
        "Survived" not in X_train.columns
    ), "Target column should be removed from features"
