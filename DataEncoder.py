import pandas as pd
from sklearn.preprocessing import LabelEncoder


def is_numerical(series):
    """Check if a pandas series is numerical."""
    return series.dtype.kind in 'biufc'


def encode_dataset(data):
    # Load dataset
    # data = pd.read_csv(csv_file)

    # Dictionary to hold LabelEncoders for each categorical column
    encoders = {}

    # Iterate over each column in the DataFrame
    for column in data.columns:
        # Check if the column is numerical
        if not is_numerical(data[column]):
            # Print unique values for non-numerical columns
            unique_values = set(data[column])
            print(f"Unique values for {column}: {unique_values}")

            # Initialize LabelEncoder for non-numerical columns
            encoder = LabelEncoder()

            # Fit and transform data for the column
            data[column] = encoder.fit_transform(data[column])

            # Store the encoder
            encoders[column] = encoder
        else:
            # Skip encoding for numerical columns
            print(f"Skipping encoding for numerical column: {column}")

    return data, encoders


def decode_dataset(encoded_data, encoders):
    # Create a copy of the data to decode
    decoded_data = encoded_data.copy()

    # Decode each encoded column using the respective encoder
    for column, encoder in encoders.items():
        decoded_data[column] = encoder.inverse_transform(encoded_data[column])

    return decoded_data


# Example usage
# encoded_data, encoders = encode_dataset('german_credit_data.csv')
# dff = pd.DataFrame(encoded_data)
#
# dff.to_csv('Encoded.csv', index=False)
#
# print(encoded_data)
# To decode, you can use
# decoded_data = decode_dataset(encoded_data, encoders)
