import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter

# ------------------- LOAD DATA -------------------
dt = pd.read_csv('Student data 4.csv')

# ------------------- FIX DUPLICATE COLUMN NAMES -------------------
def rename_duplicate_columns(columns):
    counts = Counter()
    new_cols = []
    for col in columns:
        if col == "Obtain marks":
            counts[col] += 1
            new_cols.append(f"Obtain marks {counts[col]}")
        else:
            new_cols.append(col)
    return new_cols

dt.columns = rename_duplicate_columns(dt.columns)
print("\n✅ Columns after renaming:")
print(dt.columns.tolist())

# ------------------- CLEANING -------------------
# Convert gender to numeric
dt['Gender'] = dt['Gender'].map({'male': 0, 'Male': 0, 'female': 1, 'Female': 1})

# Drop unnecessary columns
dt.drop(columns=['Inter marks'], inplace=True)
dt.drop(columns=[' ID number'], inplace=True)
print(dt.head())
# Add simulated Parent Income
income_levels = ['Low', 'Medium', 'High']
dt['Parent income'] = np.random.choice(income_levels, size=len(dt))
income_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
dt['Parent income'] = dt['Parent income'].map(income_mapping)

# ------------------- TARGET SELECTION -------------------
# Automatically find the first "Obtain marks" column
target_col = [col for col in dt.columns if "Obtain marks" in col][0]
print(f"\n🎯 Target column automatically selected: {target_col}")

X = dt.drop([target_col], axis=1)
y = dt[target_col]

# ------------------- TRAIN TEST SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- DEFINE MODELS -------------------
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

# ------------------- TRAIN & EVALUATE -------------------
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n🔹 {model_name}")
    print(f"   Mean Squared Error: {mse:.2f}")
    print(f"   R² Score: {r2:.3f}")

# ------------------- FEATURE IMPORTANCE (RANDOM FOREST) -------------------
rf_model = models['Random Forest Regressor']
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances - Random Forest Regressor")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# ------------------- CORRELATION HEATMAP -------------------
plt.figure(figsize=(10, 8))
sns.heatmap(dt.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------- MODEL COMPARISON -------------------
model_names = list(models.keys())
r2_scores = [r2_score(y_test, models[name].predict(X_test)) for name in model_names]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=r2_scores)
plt.title("Model Comparison (R² Scores)")
plt.ylabel("R² Score")
plt.show()
