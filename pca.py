'''
Performs principle component analysis on a set of train/test data
'''

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def test_pca(x_train, x_test):
    '''
    Implements sklearn packages for principle component analysis

    Input:
    x_train (pandas df): training data
    x_test (pandas df): testing data

    Returns:
    A matrix of PCA values corresponding to each variable
    '''
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    pca = PCA()
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    explained_variance = pca.explained_variance_ratio_

    return explained_variance
    