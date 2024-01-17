import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def improveOurPreviousProject():
    def get_svm_param_grid(granularity):
        # Define basic ranges
        C_range = [2**i for i in range(-5, 16, 1)]  # Powers of 2 ranging from 2^-5 to 2^15
        gamma_range = [2**i for i in range(-15, 4, 1)]  # Powers of 2 ranging from 2^-15 to 2^3

        # Adjust ranges based on granularity
        C_options = C_range[::granularity]
        gamma_options = gamma_range[::granularity]
        kernel_options = ['linear', 'rbf', 'poly', 'sigmoid']

        return {
            'C': C_options,
            'kernel': kernel_options,
            'gamma': gamma_options,
            'degree': [2, 3, 4, 5]  # Degrees for polynomial kernel
        }

    # Load a sample dataset
    df = pd.read_csv("Database.csv", index_col=0)
    for col in ['Saving accounts', 'Checking account']:
        df[col].fillna('none', inplace=True)
    j = {0: 'unskilled and non-res', 1: 'unskilled and res', 2: 'skilled', 3: 'highly skilled'}
    df['Job'] = df['Job'].map(j)

    # encoding risk as binary
    r = {"good": 0, "bad": 1}
    df['Risk'] = df['Risk'].map(r)

    # getting dummies for all the categorical variables
    dummies_columns = ['Job', 'Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account']
    for col in dummies_columns:
        df = df.merge(pd.get_dummies(df[col], drop_first=True, prefix=str(col)), left_index=True, right_index=True)

    # drop redundant variables
    columns_to_drop = ['Job', 'Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account']
    df.drop(columns_to_drop, axis=1, inplace=True)
    df['Log_CA'] = np.log(df['Credit amount'])

    X = df.drop(['Risk', 'Credit amount'], axis=1).values
    y = df['Risk'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # User input for granularity
    granularity = int(input("Enter the granularity level for SVM (1 for highest granularity, higher numbers for lower granularity): "))

    # Get parameter grid based on user-defined granularity
    param_grid = get_svm_param_grid(granularity)
    print("Parameter Grid:", param_grid)

    # Determine CV folds
    cv_folds = 10 if X_train.shape[0] < 500 else 5
    print("CV Folds:", cv_folds)

    # Initialize the SVM classifier
    model = SVC(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_folds, n_jobs=-1, verbose=2)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Evaluation
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    print(classification_report(y_test, y_pred))