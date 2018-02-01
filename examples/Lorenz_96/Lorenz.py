import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from numpy.random import uniform, normal

params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          'text.latex.unicode' : True}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams.update(params)

w = 5.33

from scipy.integrate import solve_ivp

#####      Define the model.

# --> Dimension of the model.
n = 64

# --> Forcing term.
f = 8.

def Lorenz96(t, x):

    # --> Initialize variables.
    dx = np.zeros_like(x)

    # --> First 3 edges.
    dx[0] = ( x[1] - x[n-2] ) * x[n-1] - x[0]
    dx[1] = ( x[2] - x[n-1] ) * x[0] - x[1]
    dx[n-1] = ( x[0] - x[n-3] ) * x[n-2] - x[n-1]

    # --> General case.
    for i in xrange(2, n-1):
        dx[i] = ( x[i+1] - x[i-2] ) * x[i-1] - x[i]

    # --> Add forcing term.
    dx += f

    return dx

if __name__ == '__main__':

    # --> Setup simulation.
    m = 5
    t = np.arange(0, m)*0.001
    tspan = [0., t.max()]

    # --> Generate the training dataset.
    from sparse_identification.utils import derivative
    K = 100
    X, dX = np.zeros((m, n)), np.zeros((m, n))
    for k in xrange(K):
        y0 = uniform(low=-1, high=1, size=n)
        y = solve_ivp(Lorenz96, tspan, y0, t_eval=t, method='RK45')['y'].T
        dy = derivative(y, dt=0.001)
        if k == 0:
            X = y
            dX = dy
        else:
            X = np.concatenate((X, y), axis=0)
            dX = np.concatenate((dX, dy), axis=0)

    # --> Generate the library of quadratic polynomials.
    from sklearn.preprocessing import PolynomialFeatures
    library = PolynomialFeatures(degree=2)
    A = library.fit_transform(X)

    print A.shape

    from sparse_identification import sindy
    estimator_1 = sindy(l1=0.8, solver='lasso')
    estimator_1.fit(A, dX[:, 34])

    estimator_2 = sindy(l1=0.01, solver='lstsq')
    estimator_2.fit(A, dX[:, 34])

    fig, axes = plt.subplots(1, 2)
    axes[0].stem(estimator_1.coef_)

    axes[1].stem(estimator_2.coef_)

    plt.show()
