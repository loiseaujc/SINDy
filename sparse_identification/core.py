import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sparse_identification.utils import lsqlin





def sparse_constraints(coef, sparsity_knob):

    """

    Constructs the matrix imposing the sparsity constraints.
    The sparsity knob is based on the mean value of the non-zero
    entries of coef.

    Inputs
    ------

    coef : array-like, shape (n_features)
           Coefficients from the sparse regression.

    sparsity_knob : double precision.
                    Sparsity knob of the sindy algorithm.

    Returns
    -------

    C : two-dimensional array (n_constraints, n_features)
        Matrix imposing the sparsity constraints.

    d : array-like, shape (n_constraints)
        Vector of zeros.
    
    """

    knob = sparsity_knob*abs(coef[np.nonzero(coef)]).mean()
    ind = np.where(abs(coef) <= knob)[0]
    if len(ind) > 0:
        C = np.zeros((len(ind), len(coef)))
        d = np.zeros((len(ind), 1))
        for i in range(len(ind)):
            C[i, ind[i]] = 1
        return C, d
    else:
        return None, None




    
def combine_constraints(A, b, C, d):

    """

    This function combines the user-defined constraints and
    the sparsity constraints into one single matrix.
    
    """

    if C is None:
        return A, b
    else:
        X = np.concatenate((A, C), axis=0)
        y = np.concatenate((b, d), axis=0)
        return X, y




    
def qr_rr(C, d):

    """

    QR Rank revealing decomposition of the C matrix in order to
    remove linearly dependant constraints.
    
    """
    
    q, r, p = scipy.linalg.qr(C.T, pivoting=True)
    ind = np.where(abs(np.diag(r)) < 1e-6)
    if len(ind[0]) >= 1:
        while len(ind[0]) != 0:
            C = np.delete(C, p[ind], axis=0)
            d = np.delete(d, p[ind], axis=0)
            q, r, p = scipy.linalg.qr(C.T, pivoting=True)
            ind = np.where(abs(np.diag(r)) < 1e-6)
    return C, d






class sindy(BaseEstimator, RegressorMixin):

    def __init__(self, sparsity_knob=0.001):
        self.sparsity_knob = sparsity_knob




        
    def fit(self, A, b, constraints=None):
    
        #--> Compute the initial guess for the sparse regression.
        coef = np.linalg.lstsq(A, b)[0]

        #--> Gets the user-defined constraints.
        if constraints is not None:
            C_user = constraints[0]
            d_user = constraints[1]

        #--> Sparsity promoting least-squares.
        converged = False
        
        while converged is False:
            coef_old = coef.copy()
            
            #--> Identify/Combine the constraints (sparsity + user defined)
            C_sparse, d_sparse = sparse_constraints(coef, self.sparsity_knob)
            if constraints is not None:
                C, d = combine_constraints(C_user, d_user, C_sparse, d_sparse)
            else:
                C = C_sparse
                d = d_sparse

            if C is None:
                converged=True
            else:
                #--> QR-Rank Revealing factorizing to remove linearly dependant constraints.
                C, d = qr_rr(C, d)
                output = lsqlin(A, b, Aeq=C, beq=d)
                coef = np.asarray(output['x']).squeeze()

                #--> Sets to exact zero the machine-precision zero entries.
                ind = np.where(abs(coef)<1e-10)
                coef[ind] = 0

                #--> Check convergence.
                if (coef==coef_old).all():
                    converged=True
                else:
                    coef_old = coef.copy()

        self.coef_ = coef
        
        return self




    
    def predict(self, X):
        return np.dot(X, self.coef_)
