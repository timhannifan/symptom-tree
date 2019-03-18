'''
Pandas utility functions
'''
import process
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 1


def get_df_from_csv(path, keep_cols, d_col, d_cat_col, dummies, prefix_cols):
    '''
    Reads and processes the dataset, including encoding the dependent variable
    numerically. Uses one-hot-encoding to transform symptoms and demographic
    characteristics into dummy variables. Returns the updated dataframe,
    a dictionary of diagnosis codes mapped to diagnosis strings, and
    a dictionary containing the same information with values as keys.

    Inputs:
        path (str): csv file
        keep_cols (lst): list of columns to include in the dataframe
        d_col (str): column name for original target column
        d_cat_col (str): column name for new target column
        dummies (lst): column names for the new dummy variables
        prefx_cols (lst): list of prefixes of the new column names

    Outputs:
        df2 (dataframe): updated dataframe
        d_map (dict): dictionary mapping diagnosis strings to a unique index
        d_r (dict): dictionary mapping unique numbers to diagnosis strings
    '''
    df = process.read_and_process_data(path)
    df = df.loc[:, keep_cols]
    df2 = df.copy()
    df2, d_map, d_r = encode_df_with_dict(df2, d_col, d_cat_col)
    df2 = pd.get_dummies(df2, columns=dummies, prefix=prefix_cols)

    return (df2, d_map, d_r)


def get_x_y_df(df, target_cols, target_encode_col):
    '''
    Creates predictor and target dataframes.

    Input:
        df (pandas df): containing predictor and target columns
        target_cols (lst): list of target columns (string and encoded)
        target_encode_col (str): name of target column with diagnosis codes
    Output:
        Returns a tuple of dataframes
    '''
    x = df.drop(target_cols, axis=1)
    y = df.loc[:, target_encode_col]

    return x, y


def get_test_train(x, y):
    '''
    Splits data using scikit test_train

    Input:
        x: pandas df of predictor data
        y: pandas df of target data
    Output:
        x_train (df): dataframe
        x_test (df): dataframe
        y_train (df): dataframe
        y_test (df): dataframe
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE)

    return x_train, x_test, y_train, y_test


def encode_df_with_dict(df, target, new_col):
    '''
    Add column to df with integers for target... need to access dictionary
    later.

    Input:
        df (pandas df): containing predictor and target columns
        target (str): column name of original target
        new_col (str): new column name
    Output:
        df (df): dataframe
        codes (dict): dictionary mapping diagnosis strings to a unique index
        reverse (dict): dictionary mapping unique numbers to diagnosis strings
    '''
    targets = df[target].unique()
    codes = {}
    reverse = {}
    for key, value in enumerate(targets):
        codes[value] = key
        reverse[key] = value
        df.loc[:, new_col] = df[target].replace(codes)

    return (df, codes, reverse)
