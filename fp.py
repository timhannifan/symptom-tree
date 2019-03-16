'''
Main program
'''

import pca
import process
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

DIAGNOSIS_COL = 'DIAGNOSIS_SHORT_1'
DIAGNOSIS_CAT_COL = DIAGNOSIS_COL + "_CAT"
KEEP_COLS = ['VISIT_REASON_1', 'DIAGNOSIS_SHORT_1']
DUMMY_COLS = [c for c in KEEP_COLS if c != DIAGNOSIS_COL]
PREFIX_COLS = [s[:2] for s in DUMMY_COLS]
TEST_SIZE = 0.2
RANDOM_STATE = 1


def encode_diagnoses(df, target, new_col):
    '''
    Add column to df with integers for target... need to access dictionary
    later.
    Input:
        df (pandas df): containing predictor and target columns
    Output:
        Returns a tuple of dataframes
    '''

    targets = df[target].unique()
    codes = {}
    reverse = {}
    for key, value in enumerate(targets):
        codes[value] = key
        reverse[key] = value
        df[new_col] = df[target].replace(codes)

    return (df, codes, reverse)


def split_attributes(df):
    '''
    Creates predictor and target dataframes.

    Input:
        df (pandas df): containing predictor and target columns
    Output:
        Returns a tuple of dataframes
    '''

    x = df.drop([DIAGNOSIS_COL, DIAGNOSIS_CAT_COL], axis=1)
    y = df.loc[:, DIAGNOSIS_CAT_COL]

    return x, y


def go():
    '''
    Main program process.

    Output:
        Returns a tuple containing the trained model and diagnosis map
    '''
    df = process.read_and_process_data()
    df = df.loc[:, KEEP_COLS]
    df2 = df.copy()
    df2, d_map, d_r = encode_diagnoses(df, DIAGNOSIS_COL, DIAGNOSIS_CAT_COL)
    df2 = pd.get_dummies(df2, columns=DUMMY_COLS, prefix=PREFIX_COLS)

    x, y = split_attributes(df2)
    x_train, x_test, y_train, y_test = split_data(x, y)
    trained = model(x_train, y_train)
    y_pred = trained.predict(x_test)
    # test_pca(x_train, x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    return (trained, d_map)


def split_data(x, y):
    '''
    Splits data using scikit test_train 

    Input:
        x: pandas df of predictor data
        y: pandas df of target data
    Output:
        Returns a tuple containing the trained model and diagnosis map
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE)

    return x_train, x_test, y_train, y_test


def model(x_train, y_train):
    '''
    Creates scikit decision tree, then fits the model to the training data

    Input:
        x_train: pandas df of predictor data
        y_train: pandas df of target data
    Output:
        Returns a trained DecisionTreeClassifier
    '''
    obj = DecisionTreeClassifier()
    trained_model = obj.fit(x_train, y_train)

    return trained_model


if __name__ == "__main__":
    go()
