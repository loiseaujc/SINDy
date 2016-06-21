import numpy as np

def derivative(X, dt=1.) :
    
    """ 
    
    Compute time-derivative of the data matrix X along first axis.
    
    Parameters
    ----------
    
    X : array-like, shape (n_samples,n_features)
        Input variables to be derived.
    
    dt : double precision.
         Time step.
         
    Returns
    -------
    
    dX : array-like, shape(n_samples,n_features)
         Time-derivative of matrix X.
    
    """
    
    dX = np.zeros_like(X)
    
    # Second-order accurate derivative
    
    dX[1:-1,:] = (X[2:,:]-X[:-2,:])/(2.*dt)
    
    # Treats the boundary points with a first-order derivation
    
    dX[0,:]    = (X[1,:] - X[0,:])/dt
    dX[-1,:]   = (X[-1,:] - X[-2,:])/dt
    
    return dX
