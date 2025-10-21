# src/model/train.py
import numpy as np
import pandas as pd
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model.model import build_model
import json

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '../../artifacts')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_processed():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv')).values
    X_val = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_val.csv')).values
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv')).values
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv')).values.ravel()
    y_val = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_val.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).values.ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test

def train():
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed()
    input_dim = X_train.shape[1]
    model = build_model(input_dim)

    ckpt_path = os.path.join(MODEL_DIR, 'best_model.keras')
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1)
    ]

    # Use class weights to handle imbalance
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = dict(enumerate(class_weights))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks,
        class_weight=cw
    )

    # Save training history
    with open(os.path.join(ARTIFACTS_DIR, 'train_history.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k,v in history.history.items()}, f)

    print("Training complete. Best model saved to:", ckpt_path)

if __name__ == "__main__":
    train()

"""
Why:
	•	A small NN demonstrates TensorFlow usage — layers, batchnorm, dropout.
	•	Use val_auc as early stop metric — AUC is robust for imbalanced binary classification.
	•	Class weights prevent the model from biasing toward the majority class.
"""

"""
In TensorFlow / Keras 3.x, the HDF5 format (.h5) is deprecated for saving models using model.save().
Now, Keras requires .keras as the extension for its new Keras v3 SavedModel format.
"""