import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_outliers_with_isolation_forest(df):
    """
    Detect outliers using the Isolation Forest method.
    """
    iso_forest = IsolationForest(contamination=0.05)  # contamination is the expected proportion of outliers
    preds = iso_forest.fit_predict(df)

    # Mark outliers with -1
    df['outlier'] = preds
    return df


def remove_outliers(df):
    """
    Process the CSV file to detect and remove outliers.
    """
    # Read the CSV file
    # df = pd.read_csv(input_csv)

    # Assuming all columns are relevant for outlier detection
    numeric_cols = df.select_dtypes(include='number').columns
    df_outliers_detected = detect_outliers_with_isolation_forest(df[numeric_cols])

    # Remove rows marked as outliers
    df_no_outliers = df_outliers_detected[df_outliers_detected['outlier'] != -1]

    # Write the processed data to a new CSV file
    # df_no_outliers.to_csv(output_csv, index=False)
    return df_no_outliers

# Example usage
# df = pd.read_csv('Encoded.csv')
# remove_outliers(df)
