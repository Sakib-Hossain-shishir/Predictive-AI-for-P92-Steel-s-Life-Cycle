import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

# Example dataset (replace with actual data)
data = pd.DataFrame({
    'temperature': [20, 20, 550, 600, 600, 600, 600, 550, 550, 550, 650, 650],
    'strain_amplitude': [0.4, 0.6, 0.25, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.75, 0.6, 0.8],
    'tensile_hold_time': [0, 0, 0, 0, 0, 1, 3, 10, 60, 1, 0, 0],
    'compressive_hold_time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'cycles_to_failure': [5640, 667, 9599, 9078, 4861, 1303, 1190, 675, 416, 652, 447, 381]
})

# Features and target
X = data[['temperature', 'strain_amplitude', 'tensile_hold_time', 'compressive_hold_time']]
y = data['cycles_to_failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Random Forest model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Use the best model
best_rf_model = grid_search.best_estimator_

# Evaluate model performance
cross_val_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation Mean Squared Error: {-cross_val_scores.mean()}")

# Save the best model and the scaler
with open('best_life_cycle_model.pkl', 'wb') as file:
    pickle.dump(best_rf_model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Best model and scaler saved successfully!")
