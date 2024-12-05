"""
An implemementation of the infinite SVM model with uniform weights.
"""

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from cvxpy import *


class InfSVM2(BaseEstimator, ClassifierMixin):
    """
    Implementation of an infinite SVM model with uniform weights.
    """
    
    def __init__(self, t = 1):
        """
        Parameters:
        ----------
            t: float
                The threshold to use.
        """
        self.coef_      = None
        self.intercept_ = None
        self._t         = t

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns
        -------
        self : object
            An instance of the estimator.
        """
        (n, d) = X.shape
        y      = 2 * y-1

        g = -np.ones(shape=(n, 1))

        c = np.zeros(shape=(d+1+n, 1))
        c[-n:] = 1

        A = np.zeros(shape=(n, d+1+n))

        Y = np.tile(y, reps=[d, 1]).T

        # Hinge loss's constraint
        A[:, :d]     = (-Y) * X
        A[:, d]      = -y
        A[:, d + 1:] = -np.eye(n)

        lb = np.array([None] * (d+1+n))
        ub = np.array([None] * (d+1+n))

        lb[0:d]  = -self._t
        lb[d+1:] = 0
        ub[0:d]  = self._t

        res             = scipy.optimize.linprog(c, A_ub=A, b_ub=g, bounds=np.vstack((lb,ub)).T)
        self.coef_      = res.x[0:d]
        self.intercept_ = res.x[d]
       
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or
            (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        scores = self.decision_function(X)
        
        return np.where(scores > 0, 1, 0)

    def decision_function(self, X):
        """
        Signed distance to the separating hyperplane.

        Signed distance is positive for an inlier and negative for an outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        dec : ndarray of shape (n_samples,)
            Returns the decision function of the samples.
        """
        scores = np.dot(X, self.coef_) + self.intercept_
        
        #return np.array([-scores.T, scores.T]).T
        return np.array([scores.T]).T