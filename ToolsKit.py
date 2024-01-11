from Imputer_FillMissingCSV import *
from DataEncoder import *
from RobustScaler import *

def UseToolKit(CheckBoxes = [1,1,1,1,1],CSV_path='german_credit_data.csv',missing_values_representation='NA'):

    ChangedCSV= pd.read_csv(CSV_path, na_values=missing_values_representation)

    if CheckBoxes[0]: # complete missing valuse in the csv file
        ChangedCSV = impute_csv_file(ChangedCSV)
    if CheckBoxes[1]:
        ChangedCSV, encoders = encode_dataset('german_credit_data.csv')
    if CheckBoxes[2]:
        ChangedCSV, scaler = scale_csv(ChangedCSV)
    if CheckBoxes[3]:
        print('funcion to use feature selection with grid search')
        #remember that we need to put attention on the AI model that the user chose
        #Return the model, best paramters that were chosen, the presicion.
    if CheckBoxes[4]:
        print('funcion to use feature Remove Outliers with grid search')

def IF_NEEDED_FUNCION_FOR_LVLS_4_AND_5(Temp):
    print(Temp)