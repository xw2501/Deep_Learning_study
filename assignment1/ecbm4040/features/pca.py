import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################

    dim = X.shape[1]
    eigenvalue, eigenvector = np.linalg.eig(X.T.dot(X))
    eigenvector = eigenvector.T
    
    P = np.zeros([K, dim])
    T = np.zeros(K)
    
    for i in range(K):
        P[i, :] = eigenvector[i, :]
        T[i] = np.sum(X.dot(P[i, :].T))
    
    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)
