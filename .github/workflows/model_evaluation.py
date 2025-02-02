# model_evaluation.py
from sklearn.metrics import accuracy_score, classification_report
import joblib

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

def load_model(file_path):
    return joblib.load(file_path)