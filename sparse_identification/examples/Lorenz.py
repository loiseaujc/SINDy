#######################################################################
#####                                                             #####
#####     SPARSE IDENTIFICATION OF NONLINEAR DYNAMICS (SINDy)     #####
#####     Application to the Lorenz system                        #####
#####                                                             #####
#######################################################################

"""

This small example illustrates the identification of a nonlinear
dynamical system using the data-driven approach SINDy with constraints
by Loiseau & Brunton (submitted to JFM Rapids).

Note: The sklearn python package is required for this example.
----

Contact: loiseau@mech.kth.se

"""


#--> Import standard python libraries
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#--> Import some features of scipy to simulate the systems
#    or for matrix manipulation.
from scipy.integrate import odeint
from scipy.linalg import block_diag

#--> Import the PolynomialFeatures function from the sklearn
#    package to easily create the library of candidate functions
#    that will be used in the sparse regression problem.
from sklearn.preprocessing import PolynomialFeatures

#--> Import the sparse identification python package containing
#    the class to create sindy estimators.
import sparse_identification as sp

#--> Defines various functions used in this script.

def Lorenz(x0, sigma, rho, beta, time):

    """

    This small function runs a simulation of the Lorenz system.

    Inputs
    ------

    x0 : numpy array containing the initial condition.

    sigma, rho, beta : parameters of the Lorenz system.

    time : numpy array for the evaluation of the state of
           the Lorenz system at some given time instants.

    Outputs
    -------

    x : numpy two-dimensional array.
        State vector of the vector for the time instants
        specified in time.

    xdot : corresponding derivatives evaluated using
           central differences.
    
    """

    def dynamical_system(y,t):

        dy = np.zeros_like(y)
        dy[0] = sigma*(y[1]-y[0])
        dy[1] = y[0]*(rho - y[2]) - y[1]
        dy[2] = y[0]*y[1] - beta*y[2]
        
        return dy
    
    x = odeint(dynamical_system,x0,time)
    dt = time[1]-time[0]
    xdot = sp.utils.derivative(x,dt)

    return x, xdot

def constraints(library):

    """

    This function illustrates how to impose some
    user-defined constraints for the sparse identification.

    Input
    -----

    library : library object used for the sparse identification.

    Outputs
    -------

    C : two-dimensional numpy array.
        Constraints to be imposed on the regression coefficients.

    d : one-dimensional numpy array.
        Value of the constraints.

    """

    #--> Recover the number of input and output features of the library.
    m = library.n_input_features_
    n = library.n_output_features_

    #--> Initialise the user-defined constraints matrix and vector.
    #    In this example, two different constraints are imposed.
    C = np.zeros((2, m*n))
    d = np.zeros((2,1))

    #--> Definition of the first constraint:
    #    In the x-equation, one imposes that xi[2] = -xi[1]
    #    Note: xi[0] corresponds to the bias, xi[1] to the coefficient
    #    for x(t) and xi[2] to the one for y(t).
    C[0, 1] = 1
    C[0, 2] = 1

    #--> Definition of the second constraint:
    #    In the y-equation, one imposes that xi[1] = 28
    #    Note: the n+ is because the coefficient xi[1] for
    #    the y-equation is the n+1th entry of the regression
    #    coefficients vector.
    C[1, n+1] = 1
    d[1] = 28

    return C, d

def Identified_Model(y, t, library, estimator) :
        
    '''
    Simulates the model from Sparse identification.

    Inputs
    ------

    library: library object used in the sparse identification
             (e.g. poly_lib = PolynomialFeatures(degree=3) )

    estimator: estimator object obtained from the sparse identification

    Output
    ------

    dy : numpy array object containing the derivatives evaluated using the
         model identified from sparse regression.
         
    '''

    dy = np.zeros_like(y)
        
    lib = library.fit_transform(y.reshape(1,-1))
    Theta = block_diag(lib, lib, lib)
        
    dy = np.dot(Theta, estimator.coef_)
        
    return dy

def plot_results(t, X, Y):

    """

    Function to plot the results. No need to comment.
    
    """

    fig, ax = plt.subplots( 3 , 1 , sharex = True, figsize=(10,5) )
    
    ax[0].plot(t  , X[:,0] , label='Full simulation' )
    ax[0].plot(t , Y[:,0] , 'r', label='Identified model')
    ax[0].set_ylabel('x(t)')
    ax[0].legend(loc='upper center', bbox_to_anchor=(.5, 1.33), ncol=2, frameon=False )
    
    ax[1].plot(t, X[:,1], t ,Y[:,1], 'r')
    ax[1].set_ylabel('y(t)')
    
    ax[2].plot(t ,X[:,2] ,t ,Y[:,2] ,'r')
    ax[2].set_ylabel('z(t)')
    ax[2].set_xlabel('Time')
    ax[2].set_xlim(0, 20)
    
    fig = plt.figure(figsize=(5,5))
    
    ax = fig.gca(projection='3d')
    ax.plot(X[:,0], X[:,1], X[:,2])
    ax.plot(Y[:,0], Y[:,1], Y[:,2], 'r--')
    ax.axis('equal')

    return 





if __name__ == '__main__':

    #--> Sets the parameters for the Lorenz system.
    sigma = 10.
    rho = 28.
    beta = 8./3.
    
    t = np.linspace(0,100,10000)

    #--> Run the Lorenz system to produce the data to be used in the sparse identification.
    x0 = np.array([-8., 7., 27.])
    X, dX = Lorenz(x0, sigma, rho, beta, t)

    #-----------------------------------------------------------------------
    #-----                                                             -----
    #-----     Sparse Identification of Nonlinear Dynamics (SINDY)     -----
    #-----                                                             -----
    #-----------------------------------------------------------------------
    
    # ---> Compute the Library of polynomial features.
    poly_lib = PolynomialFeatures(degree=3, include_bias=True)
    lib = poly_lib.fit_transform(X)
    Theta = block_diag(lib, lib, lib)
    n_lib = poly_lib.n_output_features_
    
    # ---> Specify the user-defined constraints.  
    C, d = constraints(poly_lib)
    
    # ---> Create a linear regression estimator using sindy and identify the underlying equations.
    estimator = sp.sindy(sparsity_knob=0.01)
    estimator.fit(Theta, dX.flatten(order='F'), constraints=[C, d])

    print "\n R2-score of the model : \n", estimator.score(Theta, dX.flatten(order='F')), "\n"
    print "\n -------------------- \n"
    print "\n \n \n Identified equation for x : \n"
    print estimator.coef_[:n_lib], "\n"
    print "\n \n \n Identified equation for y : \n"
    print estimator.coef_[n_lib:2*n_lib], "\n"
    print "\n \n \n Identified equation for z : \n"
    print estimator.coef_[2*n_lib:3*n_lib], "\n"
    print "\n -------------------- \n"
    
    #--> Simulates the identified model.
    Y  = odeint(Identified_Model, x0, t, args=(poly_lib, estimator))
    
    #--> Plots the results to compare the dynamics of the identified system against the original one.
    plot_results(t, X, Y)
    plt.show()
    
