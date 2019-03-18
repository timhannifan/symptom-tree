'''
A syptom/diagnosis decision tree based on XXX data, built with sklearn.

Authors: Tammy Glazer, Tim Hannifan, James Jensen (alpabetically)
License: MIT
'''
import graphviz
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import pdutil
import pca

DIAGNOSIS_COL = 'DIAGNOSIS_SHORT_1'
DIAGNOSIS_CAT_COL = DIAGNOSIS_COL + "_CAT"
KEEP_COLS = ['KEY', 'SEX', 'AGE_CAT', 'RACE_ETHNICITY', 'DIAGNOSIS_SHORT_1']
DUMMY_COLS = [c for c in KEEP_COLS if c != DIAGNOSIS_COL]
PREFIX_COLS = [s[:2] for s in DUMMY_COLS]


class SymptomTree:
    '''
    Class for representing the symptom/diagnosis decision tree
    '''
    def __init__(self, data):
        self.model = tree.DecisionTreeClassifier()
        self.trained_model = None
        self.data = data[0]
        self.diagnosis_dict = data[1]
        self.rev_diagnosis_dict = data[2]
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_hat = None
        self.lookup = None
        self.sym_list = []

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

    def predict(self, param=None):
        '''
        Runs a prediciton on the trained model, either from the testing data
        or from a df with similar dimensions to the x_train data

        Input:
            param: either None to use the testing data, or a pandas df
            with the same cols as x_train and a single row of values
        Output:
            either None, if no param provided, or an int corresponding
            to the diagnosis
        '''
        if param is None:
            self.y_hat = self.trained_model.predict(self.x_test)
            return None
        if isinstance(param, pd.DataFrame):
            return self.trained_model.predict(param)

        return None
    def get_user_form(self, flag=True):
        '''
        Returns a list of potential symptoms for the end user to choose from
        and initializes the base symptom dictionary. Resets the base symptom
        dictionary when called after making a prediction.

        Output:
            sym_list (list): List of all possible symptoms for end user
            to choose from
        '''

        symptoms = self.x_train.iloc[0]
        self.sym_list = list(symptoms.index)
        self.lookup = {key:0 for key in self.sym_list}

        if flag:
            return self.sym_list

        return None


    def predict_user_diagnosis(self, sym_list):
        '''
        Predicts a diagnosis given a list of symptoms, informs the user if
        they've entered an invalid symptom and resets the base
        dictionary of symptoms after each prediction.

        Inputs:
            sym_list (list): A list of symptoms

        Output
            diagnosis (str): A string that represents the predicted diagnosis
        '''
        if self.lookup is None:
            self.get_user_form()

        for sym in sym_list:
            if sym in self.lookup:
                self.lookup[sym] = 1
            else:
                print("%s is not a valid symptom" % sym)

        dataframe = pd.DataFrame.from_dict(self.lookup, orient="index")
        dataframe.reset_index(inplace=True)
        dataframe.columns = ['symptoms', 'yesno']
        row = dataframe.pivot_table(values='yesno', columns='symptoms')
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


def buildtree(raw_path):
    '''
    Main call to read data, create and train SymptomTree

    Input:
        raw_path (str): path of raw data csv

    Output:
        Returns the trained SymptomTree class object
    '''
    pd.options.mode.chained_assignment = None

    data = pdutil.get_df_from_csv(raw_path, KEEP_COLS, DIAGNOSIS_COL,
                                  DIAGNOSIS_CAT_COL, DUMMY_COLS, PREFIX_COLS)
    symptom_tree = SymptomTree(data)

    x_train, y_train = pdutil.get_x_y_df( \
        symptom_tree.data, [DIAGNOSIS_COL, DIAGNOSIS_CAT_COL],
        DIAGNOSIS_CAT_COL)

    symptom_tree.train(x_train, y_train)
    symptom_tree.predict(None)

    return symptom_tree


if __name__ == "__main__":
    PATH = 'data/symptom-tree-data.csv.csv'
    buildtree(PATH)
