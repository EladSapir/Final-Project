import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np


def is_numeric(column):
    """
    Check if a column is numeric.
    """
    return pd.api.types.is_numeric_dtype(column)


def impute_column(column, strategy, fill_value=None):
    """
    Impute a column based on the specified strategy.
    """
    if strategy == 'constant':
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy)

    return imputer.fit_transform(column.values.reshape(-1, 1))


def user_choice_for_column(column, column_name):
    """
    Prompt the user for the imputation strategy for a column based on its data type.
    """
    print(f"\nColumn: {column_name}")

    if is_numeric(column):
        print("Choose the imputation strategy for numeric data:")
        print("1: Average (mean)")
        print("2: Median")
        print("3: Most Frequent")
        print("4: Constant Value")
    else:
        print("Choose the imputation strategy for non-numeric data:")
        print("1: Most Frequent")
        print("2: Constant Value")

    choice = input("Enter your choice: ")
    if is_numeric(column):
        if choice == '1':
            return 'mean'
        elif choice == '2':
            return 'median'
        elif choice == '3':
            return 'most_frequent'
        elif choice == '4':
            constant_value = input("Enter the constant value: ")
            return ('constant', constant_value)
        else:
            print("Invalid choice. Defaulting to 'most_frequent'")
            return 'most_frequent'
    else:
        if choice == '1':
            return 'most_frequent'
        elif choice == '2':
            constant_value = input("Enter the constant value: ")
            return ('constant', constant_value)
        else:
            print("Invalid choice. Defaulting to 'most_frequent'")
            return 'most_frequent'


def impute_csv_file(file_path, missing_values):
    """
    Impute missing values in a CSV file based on user choices.
    """
    df = pd.read_csv(file_path, na_values=missing_values)

    for column in df.columns:
        strategy = user_choice_for_column(df[column], column)

        if isinstance(strategy, tuple) and strategy[0] == 'constant':
            df[column] = df[column].fillna(strategy[1])
        else:
            df[column] = impute_column(df[[column]], strategy).ravel()

    return df


# Example usage
file_path = input("Enter the path to your CSV file: ")
missing_values = input("Enter the representation of missing values in your CSV (e.g., 'NA', empty space): ")
imputed_df = impute_csv_file(file_path, missing_values)

# Save the imputed dataframe to a new CSV file
output_file_path = file_path[:-4] + '_imputed.csv'  # Change this if you want a different naming convention
imputed_df.to_csv(output_file_path, index=False)
print(f"Imputed DataFrame saved to {output_file_path}")
