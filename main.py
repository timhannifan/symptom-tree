'''
A syptom/diagnosis decision tree based on XXX data, built with sklearn.

Authors: Tammy Glazer, Tim Hannifan, James Jensen (alpabetically)
License: MIT
'''

import process
import pca
from sklearn import tree
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import pdutil
import graphviz

DIAGNOSIS_COL = 'DIAGNOSIS_SHORT_1'
DIAGNOSIS_CAT_COL = DIAGNOSIS_COL + "_CAT"
KEEP_COLS = ['KEY', 'SEX','AGE_CAT', 'RACE_ETHNICITY', 'DIAGNOSIS_SHORT_1']
DUMMY_COLS = [c for c in KEEP_COLS if c != DIAGNOSIS_COL]
PREFIX_COLS = [s[:2] for s in DUMMY_COLS]


class SymptomTree:
    def __init__(self, data):
        self.model = tree.DecisionTreeClassifier()
        self.data = data[0]
        self.diagnosis_dict = data[1]
        self.rev_diagnosis_dict = data[2]
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_hat = None

    def train(self, x_data, y_data):
        self.x_train, self.x_test, \
        self.y_train, self.y_test = pdutil.get_test_train(x_data, y_data)
        self.trained_model = self.model.fit(self.x_train, self.y_train)

    def predict(self, param):
        # pass none to use testing data
        if param is None:
            self.y_hat = self.trained_model.predict(self.x_test)
            return None
        elif isinstance(param, str):
            pass
        else:
            pass

    def get_diagnosis_string(self, code):
        try:
            diagnosis = self.rev_diagnosis_dict[code]
            return diagnosis
        except:
            return None

    def get_diagnosis_code(self, string):
        try:
            code = self.diagnosis_dict[string]
            return code
        except:
            return None

    def test_pca(self):
        res = pca.test_pca(self.x_train, self.x_test)
        print("PCA Results:", res)
        return res

    def get_diagnosis_code(self, string):
        try:
            code = self.diagnosis_dict[string]
            return code
        except:
            return None

    def visualize(self, path_fname_prefix):
        '''
        Exports a pdf visualization of the tree
        Inputs: 
            path_fname_prefix: path and filename prefix
        Outputs:
            returns nothing. exports pdf to path_fname.pdf
        '''
        dot_data = tree.export_graphviz(self.model, out_file=None) 
        graph = graphviz.Source(dot_data) 
        graph.render(path_fname_prefix) 

    @property
    def accuracy(self):
        score = accuracy_score(self.y_test, self.y_hat)
        print("Accuracy:", score)
        return score

    @property
    def predictor_set_size(self):
        return len(self.x_train.columns) - 1


def go(raw_path):
    data = pdutil.get_df_from_csv(raw_path, KEEP_COLS, DIAGNOSIS_COL,DIAGNOSIS_CAT_COL, DUMMY_COLS, PREFIX_COLS)
    st = SymptomTree(data)

    x, y = pdutil.get_x_y_df(st.data, [DIAGNOSIS_COL, DIAGNOSIS_CAT_COL],
                             DIAGNOSIS_CAT_COL)

    st.train(x, y)
    st.predict(None)

    return st


if __name__ == "__main__":
    path = 'data/d_small.csv'
    go(path)
