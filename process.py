'''
Reads and cleans data
'''

import pandas as pd
import numpy as np
import json

def get_diagnosis_map():
    with open("data/icd_codes.json", "r") as read_file:
        diagnoses = json.load(read_file)

    return diagnoses

def read_and_process_data():
    '''
    Reads in the input_data into the buffer of the decompressor.

    Input:
    input_data (string): the data into insert into the buffer

    '''
    input_fname = 'data/d_small.csv'
    output_fname = 'cleaned_small.csv'
    replacement_dict = {'AGE': {
                            "92 years or older": "92",
                            "Under 1 year": "0"},
                        'TOBACCO': {"3": np.nan},
                        'INJURY': {"2": np.nan},
                        'VISIT_REASON_CAT': {"6": "Preventive care"},
                        'HEIGHT_INCHES': {
                            "72 inches (capped value for females)": "72",
                            "77 inches (capped value for males": "77"},
                        'WEIGHT_POUNDS': {"350 lbs. or more": "350"},
                        'STATE': {"71": np.nan, "72": np.nan, "73": np.nan,
                                "74": np.nan, "96": "ESC_Div_Remainder",
                                "97": "WSC_Div_Remainder"}}

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
                 'WEIGHT_POUNDS', 'TEMP_FAHRENHEIT', 'REGION',
                 'CENSUS_DIVISION','STATE']

    bad_diagnoses = ["V990", "V991", "V992", "V997", "-9", "V99","V97",np.nan]
    bad_injury = ["Yes"]
    bad_symptoms = ["Blank"]

    df = pd.read_csv(input_fname, header=0, names=col_names, dtype=str)

    df.query("DIAGNOSIS_LONG_1 not in @bad_diagnoses", inplace=True)
    df.query("DIAGNOSIS_SHORT_1 not in @bad_diagnoses", inplace=True)
    df.query("VISIT_REASON_1 not in @bad_symptoms", inplace=True)
    df.query("INJURY not in @bad_injury", inplace=True)
    df = df[~df["VISIT_REASON_1"].str.contains('examination')]
    df.fillna(np.nan, inplace=True)
    df.replace({"-9": np.nan, "Blank": np.nan}, inplace=True)
    df.replace(replacement_dict, inplace=True)
    # df = df.loc[:, KEEP_COLS]

    dm = get_diagnosis_map()
    df['DIAGNOSIS_SHORT_1'] = df['DIAGNOSIS_SHORT_1'].apply(
        lambda x: str(x).strip('-'))
    df['DIAGNOSIS_SHORT_1'] = df['DIAGNOSIS_SHORT_1'].apply(lambda x: dm[x])

    df.to_csv(output_fname, index=False)

    return df

