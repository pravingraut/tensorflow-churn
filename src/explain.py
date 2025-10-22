import shap
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="The structure of `inputs`")

# Load model and preprocessor
model = tf.keras.models.load_model("../models/best_model.keras")
scaler = joblib.load("../artifacts/preprocessor.joblib")

# Load sample data
data = pd.read_csv("../data/processed/X_train.csv")
# X = data.drop(["Churn"], axis=1)
# X_scaled = scaler.transform(X)
# Optional: use only a sample (SHAP is computationally heavy)
X_sample = data[:100]
X_sample.shape[1]

#print("Model input shape:", model.input_shape)
#print("Data shape:", X_sample.shape)
#type(model)
#print(model.input_names)

#X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)

# Create a SHAP explainer
#explainer = shap.Explainer(model.predict, X_tensor)
# Calculate SHAP values
#shap_values = explainer(X_tensor)

explainer = shap.Explainer(lambda x: model.predict(x), X_sample)
shap_values = explainer(X_sample)

#explainer = shap.Explainer(model.predict, X_sample)
#shap_values = explainer(X_sample)

# Summary plot (global explanation)
#shap.summary_plot(shap_values, X, show=False)
#plt.savefig("reports/shap_summary.png")
#plt.close()

import os
# Create the folder if it doesn't exist
os.makedirs("reports", exist_ok=True)

shap.summary_plot(shap_values.values, X_sample, show=False)
plt.savefig("reports/shap_summary.png")
plt.close()

# Force plot for single prediction
sample_index = 5
shap.plots.force(
    base_value=shap_values.base_values[sample_index],  # per-sample expected value
    shap_values=shap_values.values[sample_index],
    features=X_sample.iloc[sample_index, :],
    matplotlib=True,
    show=False
)
plt.savefig("reports/force_plot_sample.png")
plt.close()

print("✅ SHAP explanations saved in 'reports/' folder")