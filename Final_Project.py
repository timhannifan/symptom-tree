import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


REPLACEMENT_DICT = {'AGE': {"92 years or older": "92", "Under 1 year": "0"},
                    'TOBACCO': {"3": np.nan},
                    'INJURY': {"2": np.nan},
                    'VISIT_REASON_CAT': {"6": "Preventive care"},
                    'HEIGHT_INCHES': {"72 inches (capped value for females)":
                                    "72", "77 inches (capped value for males":
                                    "77"},
                    'WEIGHT_POUNDS': {"350 lbs. or more": "350"},
                    'STATE': {"71": np.nan, "72": np.nan, "73": np.nan,
                            "74": np.nan, "96": "ESC_Div_Remainder",
                            "97": "WSC_Div_Remainder"}}

def read_and_process_data(csv_file):

    col_names = ['AGE', 'AGE_CAT', 'SEX', 'PREGNANT', 'RACE_ETHNICITY',
                 'TOBACCO', 'INJURY', 'VISIT_REASON_1', 'VISIT_REASON_2',
                 'VISIT_REASON_3', 'VISIT_REASON_CAT', 'DIAGNOSIS_LONG_1',
                 'DIAGNOSIS_LONG_2', 'DIAGNOSIS_LONG_3', 'DIAGNOSIS_SHORT_1',
                 'DIAGNOSIS_SHORT_2', 'DIAGNOSIS_SHORT_3', 'ARTHRITIS',
                 'ASTHMA', 'CANCER', 'CEREBROVASCULAR_DIS', 'COPD', 
                 'CHRONIC_RENAL_FAIL', 'CONGESTIVE_HEART_FAIL', 'DEPRESSION',
                 'DIABETES', 'HYPERLIPIDEMIA', 'HYPERTENSION', 
                 'ISCHEMIC_HEART_DIS', 'OBESITY', 'OSTEOPOROSIS', 
                 'NO_CONDITIONS', 'NUM_CONDITIONS', 'HEIGHT_INCHES', 
                 'WEIGHT_POUNDS', 'TEMP_FAHRENHEIT', 'REGION', 'CENSUS_DIVISION',
                 'STATE']

    df = pd.read_csv(csv_file, header=0, names=col_names, dtype=str)
    df.fillna(np.nan, inplace=True)
    df.replace({"-9": np.nan, "Blank": np.nan, "Entire item blank":
        np.nan}, inplace=True)
    df.replace(REPLACEMENT_DICT, inplace=True)

    return df


def num_unique(df):

    return df.nunique().sort_values()


def count_attribute(df, column, value):

    return df[df[column] == value].count()


# def populate_diagnoses:

#### James's Code ###

def encode_diagnoses(df, target):
    '''
    Add column to df with integers for target
    '''
    targets = df[target].unique()
    codes = {}
    for key, value in enumerate(targets):
        codes[value] = key
        df['unique_diagnosis'] = df[target].replace(codes)

    return df, codes


def split_attributes(df):
    #split dataset in features and target variable
    
    attributes = ['AGE', 'RACE_ETHNICITY', 'VISIT_REASON_1']
    x = df[attributes] # Things to split on
    y = df['DIAGNOSIS_SHORT_1'] # Target variable
    return x, y


def split_data(x, y):
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    return x_train, x_test, y_train, y_test


def model(x_train, y_train):

    obj = DecisionTreeClassifier()
    trained_model = obj.fit(x_train, y_train)
    prediction = trained_model.predict(x_test)

    return prediction


# Defined constants for column names
'''
AGE = 'AGE'
AGE_CAT = 'AGER'
SEX = 'SEX'
PREGNANT = 'PREGNANT'
RACE_ETHNICITY = 'RACERETH'
TOBACCO = 'USETOBAC'
INJURY = 'INJURY'
VISIT_REASON_1 = 'RFV1'
VISIT_REASON_2 = 'RFV2'
VISIT_REASON_3 = 'RFV3'
VISIT_REASON_CAT = 'MAJOR'
DIAGNOSIS_LONG_1 = 'DIAG1'
DIAGNOSIS_LONG_2 = 'DIAG2'
DIAGNOSIS_LONG_3 = 'DIAG3'
DIAGNOSIS_SHORT_1 = 'DIAG13D'
DIAGNOSIS_SHORT_2 = 'DIAG23D'
DIAGNOSIS_SHORT_3 = 'DIAG33D'
ARTHRITIS = 'ARTHRTIS'
ASTHMA = 'ASTHMA'
CANCER = 'CANCER'
CEREBROVASCULAR_DIS = 'CEBVD'
COPD = 'COPD'
CHRONIC_RENAL_FAIL = 'CRF'
CONGESTIVE_HEART_FAIL = 'CHF'
DEPRESSION = 'DEPRN'
DIABETES = 'DIABETES'
HYPERLIPIDEMIA = 'HYPLIPID'
HYPERTENSION = 'HTN'
ISCHEMIC_HEART_DIS = 'IHD'
OBESITY = 'OBESITY'
OSTEOPOROSIS = 'OSTPRSIS'
NO_CONDITIONS = 'NOCHRON'
NUM_CONDITIONS = 'TOTCHRON'
HEIGHT_INCHES = 'HTIN'
WEIGHT_POUNDS = 'WTLB'
TEMP_FAHRENHEIT = 'TEMPF'
REGION = 'REGIONOFF'
CENSUS_DIVISION = 'DIVISIONOFF'
STATE = 'FIPSSTOFF'
'''







        






















