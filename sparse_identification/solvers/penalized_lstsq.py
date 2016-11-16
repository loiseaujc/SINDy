import numpy as np
from cvxopt import solvers, matrix, spmatrix
from scipy import sparse
from scipy.linalg import norm

from numpy.linalg import LinAlgError
import warnings

from least_squares import lstsq_solve

def is_pos_def(A):

    if np.array_equal(A, A.T):  # Test the symmetry of the matrix.
        try:
            np.linalg.cholesky(A)   # Test if it is positive definite.
            return True
        except LinAlgError:
            return False
    else:
        return False

def scipy_sparse_to_cvx_sparse(x):
    '''
    This function takes as input as SciPy sparse matrix and converts it into
    a CVX sparse one.

    Inputs:
    ------
        x : SciPy sparse matrix.

    Outputs:
    -------
        y : CVX sparse matrix.
    '''

    # --> Check that the input matrix is indeed a scipy sparse matrix.
    if sparse.issparse(x) is not True:
        raise ValueError('Input matrix is not a SciPy sparse matrix.')

    # --> Convert x to COOdinate format.
    coo = x.tocoo()

    # --> Create the corresponding cvx sparse matrix.
    y = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())

    return y

def numpy_to_cvxopt_matrix(A):
    '''
    This function takes as input a numpy/SciPy array/matrix and converts it to
    a CVX format.

    Inputs:
    ------
        A : NumPy/SciPy array or matrix.

    Outputs:
    -------
        A : Corresponding matrix in CVX format.
    '''

    # --> None case.
    if A is None:
        return None

    # --> sparse SciPy case.
    if sparse.issparse(A):
        if isinstance(A, sparse.spmatrix):
            return scipy_sparse_to_spmatrix(A)
        else:
            return A
    else:
    # --> Dense matrix or NumPy array.
        if isinstance(A, np.ndarray):
            if A.ndim == 1:
                return matrix(A, (A.shape[0], 1), 'd')
            else:
                return matrix(A, A.shape, 'd')
        else:
            return A

def lasso(A, b, l1=0, l2=0, C=None, d=None, x0=None, opts=None, tol=1e-2):

    # --> Convert the matrices to CVX formats.
    A = numpy_to_cvxopt_matrix(A)
    b = numpy_to_cvxopt_matrix(b)
    C = numpy_to_cvxopt_matrix(C)
    d = numpy_to_cvxopt_matrix(d)

    # --> Sanity checks for the inputs dimensions.
    if len(A.size) != 2:
        raise warnings.warn('The input A is not a matrix, nor a two-dimensional \
        numpy array. It is transformed into a one-dimensional column vector.')
    else:
        m, n = A.size

    if b.size[0] != m:
        raise LinAlgError('A and b do not have the same number of rows.')

    if C is not None and d is None:
        raise ValueError('Matrix C has been given but not vector d. Please provide d.')
    if C is None and d is not None:
        raise ValueError('Vector d has been given but not matrix C. Please provide C.')

    if C is not None:
        k, l = C.size
        if l != n:
            raise LinAlgError('A and C do not have the same number of columns.')

        if d.size[0] != k:
            raise LinAlgError('C and d do not have the same number of rows.')

    # --> Check whether A is sparse or not.
    if sparse.issparse(A):
        sparse_case = True
    else:
        sparse_case = False

    # --> Sets up the problem.
    if m != n:
        P = A.T * A
        q = -A.T * b
    else:
        # Test if matrix is symmetric positive definite.
        spd = is_pos_def(A)
        if spd is True:
            P = A
            q = -b
        else:
            P = A.T * A
            q = -A.T * b

    # --> Ridge regularization if needed.
    if l2 < 0:
        raise ValueError('The l2-regularization weight cannot be negative.')
    if l2 > 0:
        nvars = A.size[1]
        if sparse_case is True:
            I = scipy_sparse_to_spmatrix(sparse.eye(nvars, nvars, format='coo'))
        else:
            I = matrix(np.eye(nvars), (nvars, nvars), 'd')
        P = P + l2 * I
    # x = lstsq_solve(P, -q, l2=l2, C=C, d=d, x0=x0, opts=opts)
    output = solvers.qp(P, q, None, None, C, d, x0)['x']
    x = np.asarray(output).squeeze()

    # --> Sets-up the l1-penalization problem.
    if l1 < 0:
        raise ValueError('The l1-penalization weight cannot be negative.')
    elif l1 > 1:
        raise ValueError('The l1-penalization weight cannot be larger than 1.')

    if l1 < 1:
        # --> Sets up the inequality constraint matrix for the convex optimization.
        I = matrix(0.0, (n,n))
        I[::n+1] = 1.0
        G = matrix([[I, -I, matrix(0.0, (1,n))], [-I, -I, matrix(1.0, (1,n))]])
        h = matrix(0.0, (2*n+1,1))
        h[-1] = (1.0-l1) * norm(x, ord=1)

        # --> Sets up the equlity constraint matrix.
        if C is not None:
            C_l1 = matrix(0.0, (k, 2*n))
            C_l1[:, :n] = C

        # --> Sets up the augmented problem.
        P_l1 = matrix(0.0, (2*n, 2*n))
        P_l1[:n, :n] = P
        q_l1 = matrix(0.0, (2*n, 1))
        q_l1[:n] = q

        # --> Solve the l1-penalized problem.
        if C is None:
            output = solvers.qp(P_l1, q_l1, G, h, None, None, None, None)['x'][:n]
        else:
            output = solvers.qp(P_l1, q_l1, G, h, C_l1, d, None, None)['x'][:n]
        x = np.asarray(output).squeeze()

        # -->  Get the indices of the non-zero regressors.
        for i in xrange(5):
            xmax = abs(x).max()
            I = [k for k in xrange(x.size) if abs(x[k]) > tol*xmax]
            # coef = lstsq_solve(P[I, I], -q[I], l2=l2, C=C[:, I], d=d, x0=x[I], opts=opts)
            if C is None:
                output = solvers.qp(P[I, I], q[I], None, None, None, None, x[I])['x']
            else:
                output = solvers.qp(P[I, I], q[I], None, None, C[:, I], d, x[I])['x']
            coef = np.asarray(output).squeeze()
            x[:] = 0.
            x[I] = coef

    return x
