# newtons_method.py
"""Volume 1: Newton's Method.
<Name>Natalie Larsen
<Class>001
<Date>1-29-19
"""

import numpy as np
from autograd import numpy as anp
from autograd import elementwise_grad,grad
from matplotlib import pyplot as plt
import numpy.linalg as la

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #initialize variables
    iter = 0
    xk = x0
    change = tol+1
    #perform newton's method until maxiter hit or under tolerance
    #if x is in R
    if np.isscalar(x0):
        while iter < maxiter and change > tol:
            iter += 1
            xk1 = xk
            xk = xk - alpha*f(xk)/Df(xk)
            change = abs(xk-xk1)
    #if x is in Rn
    else:
        while iter < maxiter and change > tol:
            iter += 1
            xk1 = xk
            D = Df(xk)
            #make sure the matrix isn't singular
            if la.det(D)==0:
                break
            yk = la.solve(D,f(xk))
            xk = xk - alpha*yk
            change = la.norm(xk-xk1)
    #check if method converged
    if change > tol:
        conv = False
    else:
        conv = True
    return xk,conv,iter



# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    #initialize function and its derivative
    f = lambda x: P1*((1+x)**N1-1)-P2*(1-(1+x)**-N2)
    Df = grad(f)
    #use Newton's method
    r,conv,iter = newton(f,.1,Df)
    return r


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    #initialize alphas to check
    alphas = np.linspace(.001,1,100, endpoint=True)
    results = []
    for a in alphas:
        #run newton's method for all alphas
        new = newton(f,x0,Df,tol,maxiter,a)
        results.append(list(new))
    #look at just the iterations
    iters = np.array(results)[:,2]
    #plot graph
    plt.plot(alphas,iters)
    plt.xlabel('alpha')
    plt.ylabel('iterations')
    plt.title('Newton\'s Method Comparisons')
    plt.show()
    #find index of least iterations
    smallest = np.argmin(iters)
    return alphas[smallest]


# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    #set up the points in the search interval
    x = np.linspace(-1/4,0,50)
    y = np.linspace(0,1/4,50)
    #initialize the system of function
    f = lambda x: np.array([5*x[0]*x[1]-x[0]*(1+x[1]),-x[0]*x[1]+(1-x[1])*(1+x[1])])
    #hardcode the derivative function
    Df = lambda x: np.array([[5*x[1]-1-x[1],5*x[0]-x[0]],[-x[1],-x[0]-2*x[1]]])
    #for wach point in the search interval
    for i in x:
        for j in y:
            points = np.array([i,j])
            #use newton's method with the first alpha
            a1,conv,iter = newton(f,points,Df)
            if np.all(abs(abs(a1)-np.array([0.,1.]))<np.array([.001,.001])):
                #if first passes, try ith the second alpha
                a5, conv, iter = newton(f, points, Df, alpha=.55)
                if np.all(abs(a5-np.array([3.75,.25]))<np.array([.001,.001])):
                    #return first set of points to pass both
                    return points


# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    #initialize grid
    r = np.linspace(domain[0],domain[1],res)
    i = np.linspace(domain[2],domain[3],res)
    real, imag = np.meshgrid(r,i)
    X = real + 1j*imag
    for t in range(iters):
        #run Newton's method iters times
        Xk = X
        X = Xk - f(Xk)/Df(Xk)
    Y = np.zeros((res,res))
    #for each entry, find the closest zero and store in Y
    for l in range(res):
        for k in range(res):
            Y[l,k] = np.argmin(abs(zeros - X[l,k]))
    #plot the results
    plt.pcolormesh(real, imag, Y, cmap="brg")
    plt.title('Zeros Basins')
    plt.show()
