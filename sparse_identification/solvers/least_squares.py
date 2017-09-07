import numpy as np
from cvxopt import solvers, matrix, spmatrix
from scipy import sparse
from scipy.linalg import norm

from numpy.linalg import LinAlgError
import warnings

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
    This function takes as input a numpy/SciPy array/matrix and converts it to a CVX
     format.

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

def lstsq_solve(A, b, l2=0, C=None, d=None, x0=None, opts=None):

    '''
    Basic CVX-based solver for l2-regularized linear least-squares problems.
    The corresponding minimization problem reads

        minimize   0.5 * || Ax - b ||^2_2 + l2 * || x ||^2_2
            x

        subject to  Cx = d (equality constraints)
                    Gx =< h (linear inequalities)

    This function serves as a wrapper around the CVXOPT QP solver. Note that
    the Gx <= h linear inequalities have not been implemented yet.

    Inputs:
    ------
        A  : A m x n dense or sparse matrix or numpy array.
        b  : a m x 1 vector or numpy array.
        C  : A p x n dense or sparse matrix or numpy array.
        d  : A p x 1 vector or numpy array.
        l2 : Weight for the Thikonov regularization (Ridge).

        (optional) x0 : A n x 1 vector corresponding to the initial guess.
        (optional) opts : A dictionnary of options to be passed to CVX.

    Outputs:
    -------
        output : Return dictionnary, the output of CVXOPT QP solver.
    '''
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

        if np.linalg.matrix_rank(C.T) < k:
            raise LinAlgError('C.T is rank-deficient. The svd subset selection procedure has not yet been implemented.')

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

    # --> Run the CVXOPT Quadratic Programming solver.
    #     First pass: Only the l2-penalization is accounted for.
    output = solvers.qp(P, q, None, None, C, d, None, x0)['x']
    x = np.asarray(output).squeeze()

    return x

def hard_threshold_lstsq_solve(A, b, l2=0, C=None, d=None, x0=None, opts=None, l1=0.01):

    '''
    Basic CVX-based solver for l2-regularized sequantially hard-thresholded
    least-squares problems. The corresponding minimization problem reads

        minimize   0.5 * || Ax - b ||^2_2 + l2 * || x ||^2_2
            x

        subject to  Cx = d (equality constraints)

    where sparsity of the solution vector x is imposed by sequentially
    hard-thresholding the solution. See Brunton et al. (PNAS, 2016) for more
    details.

    This function serves as a wrapper around the CVXOPT QP solver.

    Inputs:
    ------
        A  : A m x n dense or sparse matrix or numpy array.
        b  : a m x 1 vector or numpy array.
        C  : A p x n dense or sparse matrix or numpy array.
        d  : A p x 1 vector or numpy array.

        (optional) x0 : A n x 1 vector corresponding to the initial guess.
        (optional) opts : A dictionnary of options to be passed to CVX.

    Outputs:
    -------
        output : Return dictionnary, the otput of CVXOPT QP solver.
    '''
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

    # --> Run the CVXOPT Quadratic Programming solver.
    #     First pass: Only the l2-penalization is accounted for.
    output = solvers.qp(P, q, None, None, C, d, None, x0)['x']
    x = np.asarray(output).squeeze()

    # -->  Get the indices of the non-zero regressors.
    #TODO: Implement a simple convergence test to avoid unnecessary computations.
    for i in xrange(5):
        xmax = abs(x[np.nonzero(x)]).mean()
        I = [k for k in xrange(n) if abs(x[k]) > l1*xmax]
        if C is None:
            output = solvers.qp(P[I, I], q[I], None, None, None, None, x[I])['x']
        else:
            output = solvers.qp(P[I, I], q[I], None, None, C[:, I], d, x[I])['x']
        coef = np.asarray(output).squeeze()
        x[:] = 0.
        x[I] = coef

    return x

if __name__ == '__main__':
    # --> Unconstrained ordinary least-Squares example.
    A = np.array([[1, -1], [1, 1], [2, 1]])
    b = np.array([2, 4, 8])
    xref = np.array([23./7., 8./7.])
    xopt = lstsq_solve(A, b)

    if np.allclose(xopt, xref) is False:
        raise ValueError('The solution returned by the least-squares solver differs from the benchmark solution.')
