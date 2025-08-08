import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/insurance.csv")

print("--- Head of DataFrame ---")
print(df.head())

print("\n--- DataFrame Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Distribution of 'charges'
plt.figure(figsize=(10, 6))
sns.histplot(df["charges"], kde=True)
plt.title("Distribution of Medical Charges")
plt.savefig("plots/charges_distribution.png")
plt.close()

# Distribution of 'age'
plt.figure(figsize=(10, 6))
sns.histplot(df["age"], kde=True)
plt.title("Distribution of Age")
plt.savefig("plots/age_distribution.png")
plt.close()

# Distribution of 'bmi'
plt.figure(figsize=(10, 6))
sns.histplot(df["bmi"], kde=True)
plt.title("Distribution of BMI")
plt.savefig("plots/bmi_distribution.png")
plt.close()

# Categorical features: 'sex', 'smoker', 'region'
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.countplot(x="sex", data=df, ax=axes[0])
axes[0].set_title("Sex Distribution")
sns.countplot(x="smoker", data=df, ax=axes[1])
axes[1].set_title("Smoker Distribution")
sns.countplot(x="region", data=df, ax=axes[2])
axes[2].set_title("Region Distribution")
plt.tight_layout()
plt.savefig("plots/categorical_distributions.png")
plt.close()

# Relationship between 'smoker' and 'charges'
plt.figure(figsize=(8, 6))
sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Charges by Smoker Status")
plt.savefig("plots/smoker_charges.png")
plt.close()

# Relationship between 'bmi' and 'charges'
plt.figure(figsize=(10, 6))
sns.scatterplot(x="bmi", y="charges", data=df)
plt.title("Charges vs. BMI")
plt.savefig("plots/bmi_charges.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("EDA complete. Plots saved to the 'plots' directory.")


