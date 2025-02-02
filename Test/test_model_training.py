import os

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from SRC.model_training import save_model, train_model


def test_train_model():
    X_train = pd.DataFrame({"Feature1": [1, 2, 3, 4], "Feature2": [10, 20, 30, 40]})
    y_train = [0, 1, 0, 1]
    model = train_model(X_train, y_train)
    assert isinstance(
        model, RandomForestClassifier
    ), "Model should be a RandomForestClassifier"
    assert hasattr(model, "predict"), "Model should have a predict method"


def test_save_model():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    file_path = "SRC/test_model.pkl"
    save_model(model, file_path)
    assert os.path.exists(file_path), "Model file should exist"

    os.remove(file_path)
