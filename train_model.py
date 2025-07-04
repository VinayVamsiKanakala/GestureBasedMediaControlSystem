import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def calculate_features(landmarks):
    """Calculates additional features from the raw landmarks."""
    features = []
    # Normalize landmarks relative to the wrist (landmark 0)
    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
    normalized_landmarks = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in zip(landmarks[::3], landmarks[1::3], landmarks[2::3])]
    normalized_landmarks_flat = [coord for sublist in normalized_landmarks for coord in sublist]
    features.extend(normalized_landmarks_flat)

    # Calculate distances between fingertip landmarks (8, 12, 16, 20) and the wrist (0)
    finger_tip_indices = [8, 12, 16, 20]
    for finger_tip_index in finger_tip_indices:
        tip_x, tip_y, tip_z = landmarks[finger_tip_index * 3], landmarks[finger_tip_index * 3 + 1], landmarks[finger_tip_index * 3 + 2]
        distance = np.sqrt((tip_x - wrist_x)**2 + (tip_y - wrist_y)**2 + (tip_z - wrist_z)**2)
        features.append(distance)

    # Calculate distances between adjacent finger joints (example: tip to PIP)
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_index, pip_index in finger_pairs:
        tip_x, tip_y, tip_z = landmarks[tip_index * 3], landmarks[tip_index * 3 + 1], landmarks[tip_index * 3 + 2]
        pip_x, pip_y, pip_z = landmarks[pip_index * 3], landmarks[pip_index * 3 + 1], landmarks[pip_index * 3 + 2]
        distance = np.sqrt((tip_x - pip_x)**2 + (tip_y - pip_y)**2 + (tip_z - pip_z)**2)
        features.append(distance)

    return np.array(features)

# Load the data from the CSV file
CSV_FILE = 'gesture_data.csv'

if not os.path.exists(CSV_FILE):
    print(f"Error: CSV file '{CSV_FILE}' not found. Run collect_data_modified.py first.")
    exit()

try:
    data = pd.read_csv(CSV_FILE)
except pd.errors.EmptyDataError:
    print(f"Error: CSV file '{CSV_FILE}' is empty. Run collect_data_modified.py to collect data.")
    exit()

if 'label' not in data.columns:
    print(f"Error: 'label' column not found in '{CSV_FILE}'. Check data collection.")
    print("Columns found:", data.columns.tolist())
    exit()

# Separate features (landmarks) and labels (gestures)
X_raw = data.drop('label', axis=1)
y = data['label']

# Apply feature engineering
X_engineered = np.array([calculate_features(row.values) for index, row in X_raw.iterrows()])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'linear']}

# Initialize GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Train the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Hyperparameters: {best_params}")

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model's accuracy and print a classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Model: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a pickle file
MODEL_FILE = 'gesture_model.pkl'
with open(MODEL_FILE, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Trained model saved to {MODEL_FILE}")
