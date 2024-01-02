import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def get_knn_param_grid(granularity):
    # Define basic ranges
    n_neighbors_range = range(1, 31)

    # Adjust ranges based on granularity
    n_neighbors_options = n_neighbors_range[::granularity]

    return {
        'n_neighbors': list(n_neighbors_options),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }

# Load a sample dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User input for granularity
granularity = int(input("Enter the granularity level (1 for highest granularity, higher numbers for lower granularity): "))

# Get parameter grid based on user-defined granularity
param_grid = get_knn_param_grid(granularity)
print("Parameter Grid:", param_grid)

# Initialize the classifier
model = KNeighborsClassifier()

# Determine CV folds
cv_folds = 5  # Adjust as necessary

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_folds, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))
