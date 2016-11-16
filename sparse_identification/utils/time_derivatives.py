import numpy as np

def derivative(x, dt=1.) :

    """

    Compute time-derivative of the data matrix X along first axis.

    Parameters
    ----------

    X : array-like, shape (n_samples,n_features)
        Input variables.

    dt : double precision.
         Time step.

    Returns
    -------

    dX : array-like, shape(n_samples,n_features)
         Time-derivative of matrix X.

    """

    # --> Check the dimensions of X.
    if len(x.shape) != 2:
        x = x[:, np.newaxis]

    # --> Initialize the return array.
    dx = np.zeros_like(x)

    # --> Second-order accurate derivative
    dx[1:-1, :] = (x[2:, :] - x[:-2, :])/2.

    # --> Treats the boundary points with a first-order derivation
    dx[0,:]    = (x[1,:] - x[0,:])
    dx[-1,:]   = (x[-1,:] - x[-2,:])

    return dx/dt
