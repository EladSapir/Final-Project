import pandas as pd
from sklearn.preprocessing import RobustScaler

def scale_csv(df):
    # Read the CSV file
    # df = pd.read_csv(input_csv_path)

    # Initialize the RobustScaler
    scaler = RobustScaler()

    # Scale the data
    scaled_data = scaler.fit_transform(df)

    # Convert scaled data back to DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    # Return the scaled DataFrame and the scaler
    return scaled_df, scaler

def inverse_scale_csv(scaled_df, scaler):
    # Inverse transform the data
    original_data = scaler.inverse_transform(scaled_df)

    # Convert to DataFrame
    original_df = pd.DataFrame(original_data, columns=scaled_df.columns)

    # Return the DataFrame in its original scale
    return original_df

# Example usage
# input_csv = 'path/to/your/input.csv'  # Path to your input CSV file
#
# # Scale the CSV data
# scaled_df, scaler = scale_csv(input_csv)
#
# # Perform your analysis or processing with scaled_df here...
#
# # If needed, inverse scale to get the original data
# original_df = inverse_scale_csv(scaled_df, scaler)

# Now, scaled_df and original_df can be used as needed, or saved to CSVs
