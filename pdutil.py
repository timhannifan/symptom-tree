'''
Pandas utility functions
'''
import process
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 1


def get_df_from_csv(path, keep_cols, d_col, d_cat_col, dummies, prefix_cols):
    df = process.read_and_process_data(path)
    df = df.loc[:, keep_cols]
    df2 = df.copy()
    df2, d_map, d_r = encode_df_with_dict(df2, d_col, d_cat_col)
    df2 = pd.get_dummies(df2, columns=dummies, prefix=prefix_cols)

    return (df2, d_map, d_r)


def get_x_y_df(df, target_cols, target_string_col):
    '''
    Creates predictor and target dataframes.

    Input:
        df (pandas df): containing predictor and target columns
    Output:
        Returns a tuple of dataframes
    '''

    x = df.drop(target_cols, axis=1)
    y = df.loc[:, target_string_col]

    return x, y


def get_test_train(x, y):
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


def encode_df_with_dict(df, target, new_col):
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
