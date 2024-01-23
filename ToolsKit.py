from Imputer_FillMissingCSV import *
from DataEncoder import *
from RobustScaler import *
from FeatureSelection import *
from RemoveOutliers import *


def UseToolKit(CheckBoxes, CSV_path='Database.csv', missing_values_representation='NA', k=1):
    ChangedCSV = pd.read_csv(CSV_path, na_values=missing_values_representation)
    CSVsize = ChangedCSV.shape[1]
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
        FSres = UseFeatureSelection(ChangedCSV)     # function down lets to remove until there are 4 columns(1each time)
        if FSres[0]:
            ChangedCSV = FSres[1]
            History.append([ChangedCSV, None, None])
        # remember that we need to put attention on the AI model that the user chose
        # Return the model, best paramters that were chosen, the presicion.
    if CheckBoxes[4]:
        ROres = UseRemoveOutliers(ChangedCSV, CSVsize)  # function down lets to remove 10% of original CSV
        if ROres:
            ChangedCSV = ROres[1]
            History.append([ChangedCSV, None, None])

    return History


def UseFeatureSelection(ChangedCSV):
    numberOfColumns = ChangedCSV.shape[0]
    if numberOfColumns <= 4:
        return [False]
    else:
        ChangedCSV = [True, feature_selection(ChangedCSV, 1)]


def UseRemoveOutliers(ChangedCSV, CSVsize):
    numberOfRows = ChangedCSV.shape[1]
    if numberOfRows / CSVsize <= 0.9:
        return [False]
    else:
        return [True, remove_outliers(ChangedCSV)]
