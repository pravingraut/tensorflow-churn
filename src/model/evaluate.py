# src/model/evaluate.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import json

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/best_model.keras')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '../../artifacts')

def load_data():
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv')).values
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).values.ravel()
    return X_test, y_test

def evaluate():
    model = load_model(MODEL_PATH)
    X_test, y_test = load_data()
    preds = model.predict(X_test).ravel()
    pred_labels = (preds > 0.5).astype(int)

    auc = roc_auc_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred_labels, average='binary')

    cm = confusion_matrix(y_test, pred_labels).tolist()
    metrics = {
        'roc_auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm
    }

    with open(os.path.join(ARTIFACTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    evaluate()