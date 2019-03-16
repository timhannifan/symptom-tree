import numpy as np
import pandas as pd
import get_ICD
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score



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

DIAGNOSIS_COL = 'DIAGNOSIS_SHORT_1'
DIAGNOSIS_CAT_COL = DIAGNOSIS_COL + "_CAT"
KEEP_COLS = ['VISIT_REASON_1', 'DIAGNOSIS_SHORT_1']
DUMMY_COLS = [c for c in KEEP_COLS if c != DIAGNOSIS_COL]
PREFIX_COLS = [s[:2] for s in DUMMY_COLS]



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
                 'WEIGHT_POUNDS', 'TEMP_FAHRENHEIT', 'REGION', 'CENSUS_DIVISION','STATE']

    df = pd.read_csv(csv_file, header=0, names=col_names, dtype=str)
    
    bad_diagnoses = ["V990", "V991", "V992", "V997", "-9", "V99","V97"]
    df.query("DIAGNOSIS_LONG_1 not in @bad_diagnoses", inplace=True)
    df.query("DIAGNOSIS_SHORT_1 not in @bad_diagnoses", inplace=True)

    bad_symptoms = ["Blank"]
    df.query("VISIT_REASON_1 not in @bad_symptoms", inplace=True)
    df = df[~df["VISIT_REASON_1"].str.contains('examination')]

    bad_injury = ["Yes"]
    df.query("INJURY not in @bad_injury", inplace=True)
    df.fillna(np.nan, inplace=True)
    df.replace({"-9": np.nan, "Blank": np.nan}, inplace=True)
    df.replace(REPLACEMENT_DICT, inplace=True)
    df = df.loc[:, KEEP_COLS]

    # REMOVE BEFOFRE PRODUCTION
    df = df.iloc[0:100,:] 

    df['DIAGNOSIS_SHORT_1'] = df['DIAGNOSIS_SHORT_1'].apply(lambda x: x.strip('-'))
    df['DIAGNOSIS_SHORT_1'] = df['DIAGNOSIS_SHORT_1'].apply(lambda x: get_ICD.translate_general(x))

    return df

def test_pca(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    pca = PCA()
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)


def go(df): 
    df2 = df.copy()
    df2 = encode_diagnoses(df, DIAGNOSIS_COL, DIAGNOSIS_CAT_COL)
    df2 = pd.get_dummies(df2, columns=DUMMY_COLS, prefix=PREFIX_COLS)

    x, y = split_attributes(df2)
    x_train, x_test, y_train, y_test = split_data(x, y)
    trained = model(x_train, y_train)
    y_pred = trained.predict(x_test)

    test_pca(x_train, x_test)


    cm = confusion_matrix(y_test, y_pred)  
    print(cm)  
    # print('CM Accuracy' + accuracy_score(y_test, y_pred))  

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return trained


def encode_diagnoses(df, target, new_col):
    '''
    Add column to df with integers for target... need to access dictionary later
    '''
    targets = df[target].unique()
    codes = {}
    for key, value in enumerate(targets):
        codes[value] = key
        df[new_col] = df[target].replace(codes)

    return df 


def split_attributes(df):
    x = df.drop([DIAGNOSIS_COL, DIAGNOSIS_CAT_COL], axis=1)
    y = df.loc[:,DIAGNOSIS_CAT_COL]

    return x, y


def split_data(x, y):
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    return x_train, x_test, y_train, y_test


def model(x_train, y_train):
    obj = DecisionTreeClassifier()
    trained_model = obj.fit(x_train, y_train)

    return trained_model

if __name__ == "__main__":

    go(sys.argv)