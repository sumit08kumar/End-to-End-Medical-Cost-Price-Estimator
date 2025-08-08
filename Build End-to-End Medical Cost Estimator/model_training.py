import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data (assuming data_preprocessing.py has been run and saved)
df = pd.read_csv("data/insurance.csv")

# Separate target variable
X = df.drop("charges", axis=1)
y = df["charges"]

# Handle missing values in the target variable
y = y.fillna(y.mean())  # Replace NaN values in `y` with the mean

# Define categorical and numerical features
categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]

# Ensure numerical columns are numeric and handle missing values
X[numerical_features] = X[numerical_features].apply(pd.to_numeric, errors='coerce')
X[numerical_features] = X[numerical_features].fillna(X[numerical_features].mean())

# Load the preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# Transform the data
X_processed = preprocessor.transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "SVR": SVR()
}

# Train and evaluate models
results = {}
best_model = None
best_r2 = -np.inf

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

print("\nModel Evaluation Results:")
for name, metrics in results.items():
    print(f"--- {name} ---")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")

print(f"\nBest performing model: {best_model_name} with R2 Score: {best_r2:.2f}")

# Save the best model
if best_model:
    joblib.dump(best_model, "models/final_model.pkl")
    print(f"Best model ({best_model_name}) saved to models/final_model.pkl")

# SHAP Interpretability for the best model
if best_model:
    print(f"\nGenerating SHAP explanations for {best_model_name}...")
    # For tree-based models, use TreeExplainer
    if isinstance(best_model, (RandomForestRegressor, XGBRegressor)):
        explainer = shap.TreeExplainer(best_model)
        # SHAP values for the test set
        shap_values = explainer.shap_values(X_test)
    else:
        # For other models, use KernelExplainer (can be slow for large datasets)
        # It's recommended to sample a smaller background dataset for KernelExplainer
        background = shap.utils.sample(X_train, 100) # Sample 100 instances from training data
        explainer = shap.KernelExplainer(best_model.predict, background)
        shap_values = explainer.shap_values(X_test)

    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Visualize the SHAP summary plot
    # If shap_values is a list (for multi-output models), take the first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary Plot for {best_model_name}")
    plt.tight_layout()
    plt.savefig("plots/shap_summary_plot.png")
    plt.close()

print("Model training and SHAP interpretability complete.")