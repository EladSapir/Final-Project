import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def get_param_grid(granularity, n_samples):
    # Define basic ranges
    n_estimators_range = range(50, 701, 50)
    max_depth_range = range(3, 31)

    # Adjust ranges based on granularity
    n_estimators_options = n_estimators_range[::granularity]
    max_depth_options = max_depth_range[::granularity]

    if n_samples < 500:
        max_features_options = ['sqrt', 'log2']
    elif n_samples < 1000:
        max_features_options = ['sqrt', 'log2', 'auto']
    else:
        max_features_options = ['sqrt', 'log2', 'auto', None]

    return {
        'n_estimators': list(n_estimators_options),
        'max_features': max_features_options,
        'max_depth': list(max_depth_options),
        'criterion': ['gini', 'entropy']
    }

# Load a sample dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User input for granularity
granularity = int(input("Enter the granularity level (1 for highest granularity, higher numbers for lower granularity): "))

# Determine dataset size
n_samples = X_train.shape[0]

# Get parameter grid based on user-defined granularity and dataset size
param_grid = get_param_grid(granularity, n_samples)
print("Parameter Grid:", param_grid)

# Determine CV folds
cv_folds = 10 if n_samples < 500 else 5
print("CV Folds:", cv_folds)

# Initialize the classifier
model = RandomForestClassifier(random_state=42)

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
