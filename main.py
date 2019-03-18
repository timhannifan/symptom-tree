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
    '''
    Class for representing the symptom/diagnosis decision tree
    '''
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
        self.lookup = None

    def train(self, x_data, y_data):
        '''
        Trains the internal model using a set of x and y testing data

        Input:
            x_data (df): predictor variable data
            y_data (df): dependent variable data

        Output:
            Returns nothing
        '''
        self.x_train, self.x_test, \
        self.y_train, self.y_test = pdutil.get_test_train(x_data, y_data)
        self.trained_model = self.model.fit(self.x_train, self.y_train)

    def predict(self, param):
        # pass none to use testing data
        if param is None:
            self.y_hat = self.trained_model.predict(self.x_test)
            return None

        elif isinstance(param, pd.DataFrame):
            
            return self.trained_model.predict(param)

    def get_user_form(self, flag=True):
        symptoms = self.x_train.iloc[0] # want to get first fow of x_train so that we have all potential symptoms in the model
        self.sym_list = list(symptoms.index)
        self.lookup = {key:0 for key in self.sym_list}

        if flag:
            return self.sym_list
        else:
            return None


    def predict_user_diagnosis(self, sym_list):
        if self.lookup is None:
            self.get_user_form()

        for sym in sym_list:
            if sym in self.lookup:
                self.lookup[sym] = 1
            else:
                print("%s is not a valid symptom" % sym)

        df = pd.DataFrame.from_dict(self.lookup, orient="index")
        df.reset_index(inplace=True)
        df.columns = ['symptoms', 'yesno']
        row = df.pivot_table(values= 'yesno', columns='symptoms')
        code_array = self.predict(row)
        self.get_user_form(flag=False)

        return self.get_diagnosis_string(code_array[0])

    def test_pca(self):
        '''
        Calls the pca module function test_pca to report explained variance

        Input:
            none

        Output:
            Returns explained_variance matrix
        '''
        return pca.test_pca(self.x_train, self.x_test)

    def get_diagnosis_string(self, code):
        '''
        Fetches a diagnosis string given a unique diagnosis code

        Input:
            code (int): diagnosis unique identifier

        Output:
            diagnosis string (str)
        '''
        try:
            diagnosis = self.rev_diagnosis_dict[code]
            return diagnosis
        except:
            return None

    def get_diagnosis_code(self, string):
        '''
        Fetches a unique internal diagnosis code from the processed data

        Input:
            string (str): diagnosis string

        Output:
            diagnosis code (int)
        '''
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

    def print_col_name(self, idx):
        '''
        Retrieves a column name given an X index
        Inputs: 
            idx: int
        Outputs:
            string column name or None
        '''
        cols = self.data.columns

        try:
            return cols[idx]
        except:
            return None

    @property
    def accuracy(self):
        '''
        Reports the accuracy of the trained model using testing data

        Output:
            Returns the trained SymptomTree class object
        '''
        return accuracy_score(self.y_test, self.y_hat)

    @property
    def predictor_set_size(self):
        '''
        Reports size of the predictor variable set

        Output:
            Returns the trained SymptomTree class object
        '''
        return len(self.x_train.columns) - 1


def go(raw_path):
    '''
    Main call to read data, create and train SymptomTree

    Input:
        raw_path (str): path of raw data csv

    Output:
        Returns the trained SymptomTree class object
    '''
    pd.options.mode.chained_assignment = None

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
