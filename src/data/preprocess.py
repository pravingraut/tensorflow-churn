# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/telco_churn.csv')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '../../artifacts')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    return df

def clean_df(df):
    # Basic cleaning
    # Drop customerID (unique id) - not useful for modeling
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Convert 'TotalCharges' to numeric (some spaces cause conversion issue)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing numeric with median, categorical with mode (simple approach)
    for col in df.columns:
        if df[col].dtype in ['float64','int64'] and df[col].isna().sum()>0:
            df[col].fillna(df[col].median(), inplace=True)
        if df[col].dtype == 'object' and df[col].isna().sum()>0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Map target to 0/1
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    return df

def build_preprocessor(df):
    # Select features (drop target)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Identify numeric and categorical
    numeric_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object','category']).columns.tolist()

    # Example: Some binary categorical columns are 'Yes'/'No' already - keep as categorical
    # Create pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    return preprocessor, X, y, numeric_feats, categorical_feats

def split_and_save(df, preprocessor):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Fit preprocessor on train and transform train/val/test
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Save processed CSVs (optionally save transformed numpy arrays)
    pd.DataFrame(X_train_trans).to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    pd.DataFrame(X_val_trans).to_csv(os.path.join(PROCESSED_DIR, 'X_val.csv'), index=False)
    pd.DataFrame(X_test_trans).to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(PROCESSED_DIR, 'y_val.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))
    print("Saved preprocessor and processed data to folders.")

if __name__ == "__main__":
    df = load_raw()
    df = clean_df(df)
    preprocessor, X, y, num_feats, cat_feats = build_preprocessor(df)
    split_and_save(df, preprocessor)


"""
Explanation / justification:
	•	We use ColumnTransformer so numeric and categorical flows are separate and reproducible.
	•	StandardScaler stabilizes training for neural nets.
	•	OneHotEncoder(handle_unknown='ignore') avoids issues when unseen categories appear in production.
	•	We fit only on training data to avoid leakage.
	•	Save the preprocessor.joblib for production inference.
"""