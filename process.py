'''
Reads and cleans data
'''

import pandas as pd
import numpy as np
import json

DIAGNOSIS_PATH = "data/icd_codes.json"
OUTPUT_FNAME = 'cleaned.csv'

REPLACEMENT_DICT = {'AGE': {
                        "92 years or older": "92",
                        "Under 1 year": "0"},
                    'TOBACCO': {"3": np.nan},
                    'INJURY': {"2": np.nan},
                    'VISIT_REASON_CAT': {"6": "Preventive care"},
                    'HEIGHT_INCHES': {
                        "72 inches (capped value for females)": "72",
                        "77 inches (capped value for males)": "77"},
                    'WEIGHT_POUNDS': {"350 lbs. or more": "350"},
                    'STATE': {"71": np.nan, "72": np.nan, "73": np.nan,
                            "74": np.nan, "96": "ESC_Div_Remainder",
                            "97": "WSC_Div_Remainder"}}

COL_NAMES = ['AGE', 'AGE_CAT', 'SEX', 'PREGNANT', 'RACE_ETHNICITY',
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

BAD_DIAGNOSES = ["V990","V990-", "V991", "V992", "V997", "-9", "V99","V97", np.nan]
BAD_INJURY = ["Yes"]
BAD_SYMPTOMS = ["Blank", "visit", "medication", "counseling", "injury", "surgery"]
VISIT_REASON_COL = 'VISIT_REASON_1'
DIAGNOSIS_COL = 'DIAGNOSIS_SHORT_1'
LONG_NAMES = ['VISIT_REASON_1', 'SYMP1','SYMP2','SYMP3', 'SYMP4']

def get_diagnosis_map():
    '''
    Reads in the diagnosis map from json file. 

    Output:
        Returns a dictionary matching ICD codes to long-form diagnosis strings
    '''
    with open(DIAGNOSIS_PATH, "r") as read_file:
        diagnoses = json.load(read_file)

    return diagnoses


def read_and_process_data(filename):
    '''
    Reads the raw input csv data, performs cleaning, filling, and replacement
    tasks, exports a csv file of cleaned data, and returns the cleaned df

    Output:
        Returns a dictionary matching ICD codes to long-form strings
    '''

    df = pd.read_csv(filename, header=0, names=COL_NAMES, dtype=str)

    df.query("DIAGNOSIS_LONG_1 not in @BAD_DIAGNOSES", inplace=True)
    df.query("DIAGNOSIS_SHORT_1 not in @BAD_DIAGNOSES", inplace=True)
    df.query("VISIT_REASON_1 not in @BAD_SYMPTOMS", inplace=True)
    df.query("INJURY not in @BAD_INJURY", inplace=True)
    df = df[~df["VISIT_REASON_1"].str.contains('examination')]
    df.fillna(np.nan, inplace=True)
    df.replace({"-9": np.nan, "Blank": np.nan}, inplace=True)
    df.replace(REPLACEMENT_DICT, inplace=True)

    dm = get_diagnosis_map()
    df['DIAGNOSIS_SHORT_1'] = df['DIAGNOSIS_SHORT_1'].apply(
        lambda x: str(x).strip('-'))
    df['DIAGNOSIS_SHORT_1'] = df['DIAGNOSIS_SHORT_1'].apply(
        lambda x: dm[x] if x in dm else '')

    sorted_df = go_long(df)
    df = pd.merge(df, sorted_df, on=VISIT_REASON_COL, how='left')

    df[VISIT_REASON_COL] = df[VISIT_REASON_COL].str.lower()
    df[DIAGNOSIS_COL] = df[DIAGNOSIS_COL].str.lower()

    df.to_csv(OUTPUT_FNAME, index=False)

    return df


def go_long(df):

    new = df[VISIT_REASON_COL].str.split(',', expand = True)
    new_df = df[VISIT_REASON_COL].to_frame().join(new)
    col_names = ['SYMP' + str(i) for i in range(1,len(new_df.columns))]
    long_names = [VISIT_REASON_COL] + col_names
    new_df.columns = long_names

    keys = [c for c in new_df if c.startswith('SYMP')]
    melted_df = pd.melt(new_df, id_vars=VISIT_REASON_COL, value_vars=keys, value_name='KEY')
    no_dupes = melted_df.drop_duplicates()

    sorted_df = no_dupes.sort_values([VISIT_REASON_COL, 'variable'])
    sorted_df.drop('variable', axis=1, inplace=True)
    sorted_df.dropna(inplace=True)
    sorted_df = sorted_df[~sorted_df['KEY'].str.contains('...', regex=False)]
    sorted_df['KEY'] = sorted_df['KEY'].apply(
        lambda x: str(x).strip())
    sorted_df['KEY'] = sorted_df['KEY'].str.lower()

    return sorted_df
