import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sparse_identification.solvers import lstsq_solve, hard_threshold_lstsq_solve
from sparse_identification.solvers import lasso

class sindy(BaseEstimator, RegressorMixin):

    def __init__(self, l1=0, l2=1e-4, tol=1e-3, solver='lstsq'):
        # --> l1-regularization weight.
        self.l1 = l1
        # --> Thikonov regularization weight.
        self.l2 = l2
        # --> Tolerance in the LASSO solver.
        self.tol = tol
        # --> Choice of solver: lstsq or lasso.
        self.solver = solver

    def fit(self, A, b, eq=None, ineq=None):

        #--> Gets the user-defined constraints.
        if eq is not None:
            C = eq[0]
            d = eq[1]
        else:
            C = None
            d = None

        if self.solver == 'lstsq':
            #--> Compute the initial guess for the sparse regression.
            if self.l1 == 0:
                coef = lstsq_solve(A, b, C=C, d=d, l2=self.l2)
            else:
                coef = hard_threshold_lstsq_solve(A, b, C=C, d=d, l2=self.l2, l1=self.l1)
        elif self.solver == 'lasso':
            coef = lasso(A, b, C=C, d=d, l2=self.l2, l1=self.l1, tol=self.tol)
        else:
            return ValueError('Desired solver has not been implemented yet.')

        self.coef_ = coef

        return self

    def predict(self, X):
        return np.dot(X, self.coef_)
