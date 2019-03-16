import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
import json

DIAGNOSIS_COL = 'DIAGNOSIS_SHORT_1'
DIAGNOSIS_CAT_COL = DIAGNOSIS_COL + "_CAT"
KEEP_COLS = ['VISIT_REASON_1', 'DIAGNOSIS_SHORT_1']
DUMMY_COLS = [c for c in KEEP_COLS if c != DIAGNOSIS_COL]
PREFIX_COLS = [s[:2] for s in DUMMY_COLS]

def test_pca(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    pca = PCA()
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)



def go(): 
    # df = read_and_process_data('data/d.csv')
    df = pd.read_csv('cleaned.csv')
    df = df.loc[:, KEEP_COLS]
    df2 = df.copy()
    df2, d_map = encode_diagnoses(df, DIAGNOSIS_COL, DIAGNOSIS_CAT_COL)
    df2 = pd.get_dummies(df2, columns=DUMMY_COLS, prefix=PREFIX_COLS)

    x, y = split_attributes(df2)
    x_train, x_test, y_train, y_test = split_data(x, y)
    trained = model(x_train, y_train)
    y_pred = trained.predict(x_test)

    # test_pca(x_train, x_test)


    # cm = confusion_matrix(y_test, y_pred)  
    # print(cm)  

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return (trained, d_map)


def encode_diagnoses(df, target, new_col):
    '''
    Add column to df with integers for target... need to access dictionary later
    '''
    targets = df[target].unique()
    codes = {}
    for key, value in enumerate(targets):
        codes[value] = key
        df[new_col] = df[target].replace(codes)

    return (df, codes)


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
    go()