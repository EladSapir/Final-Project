from Imputer_FillMissingCSV import *
from DataEncoder import *
from RobustScaler import *
from FeatureSelection import *
from RemoveOutliers import *


def UseToolKit(CheckBoxes , CSV_path='Database.csv', missing_values_representation='NA', k=1):
    ChangedCSV = pd.read_csv(CSV_path, na_values=missing_values_representation)

    History = []  # each history is list of 3 [ data , encoders , scaler ] to be able to return, None if empty.

    if CheckBoxes[0]:  # complete missing valuse in the csv file
        ChangedCSV = impute_csv_file(ChangedCSV)
        History.append([ChangedCSV, None, None])
    if CheckBoxes[1]:
        ChangedCSV, Encoders = encode_dataset(ChangedCSV)
        History.append([ChangedCSV, Encoders, None])
    if CheckBoxes[2]:
        ChangedCSV, Scaler = scale_csv(ChangedCSV)
        History.append([ChangedCSV, None, Scaler])
    if CheckBoxes[3]:
        ChangedCSV = feature_selection(ChangedCSV,k)
        History.append([ChangedCSV, None, None])
        # remember that we need to put attention on the AI model that the user chose
        # Return the model, best paramters that were chosen, the presicion.
    if CheckBoxes[4]:
        ChangedCSV = remove_outliers(ChangedCSV)
        History.append([ChangedCSV, None, None])

    return History


def IF_NEEDED_FUNCION_FOR_LVLS_4_AND_5(Temp):
    print(Temp)
