import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def feature_selection(data, k=10):
    # Load the CSV file
    # data = pd.read_csv(csv_path)

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Apply feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Get the columns that were selected
    mask = selector.get_support()
    selected_columns = X.columns[mask]

    # Create a new DataFrame with selected features and the target
    selected_data = pd.DataFrame(X_new, columns=selected_columns)
    selected_data['target'] = y

    # # Save the new DataFrame to a CSV file
    # selected_data.to_csv('selected_features.csv', index=False)

    return selected_data

# Example usage
# csv_path = pd.read_csv('db.csv')
# result_csv = feature_selection(csv_path, k=3)
# print(f"Feature selected data saved to: {result_csv}")
