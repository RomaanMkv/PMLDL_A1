import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.preprocess import preprocess_data


# Load and preprocess data
raw_data = pd.read_csv('data/flat-prices.csv')
raw_data = raw_data.head(100)
print("Loaded raw data from 'data/flat-prices.csv'")

prep_path = 'data/prep.pkl'
X, y = preprocess_data(data=raw_data, prep_path=prep_path)
y = np.array(y).ravel()
print("Data preprocessing completed")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets")

# Define the model
model = GradientBoostingRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
print("GridSearchCV setup complete")

# Fit the model
grid_search.fit(X_train, y_train)
print("Model training complete")

# Get the best model
best_model = grid_search.best_estimator_
print("Best model identified")

# Save the best model
artifact_path = "models/basic_gb.pkl"
joblib.dump(best_model, artifact_path)
print(f"Model saved to {artifact_path}")

# Evaluate the model
test_score = best_model.score(X_test, y_test)
print(f"Test R^2 score: {test_score}")

# To load the model later
# loaded_model = joblib.load(artifact_path)
