# solutions.py
"""Volume 2: Gaussian Quadrature.
<Name> Natalie Larsen
<Class> 001
<Date> 1-24-19
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eig
from scipy.sparse import diags
from scipy.stats import norm
from scipy.integrate import quad


class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        if polytype == 'chebyshev':
            #store polytype as chebyshev with accompanying inverse weights
            self.polytype = polytype
            self.inv_w = lambda x: np.sqrt(1-x**2)
        elif polytype == 'legendre':
            #store polytype as legendre with accompanying inverse weights
            self.polytype = polytype
            self.inv_w = lambda x: 1
        else:
            #error for invalid polytype
            raise ValueError('Must be polytype legendre or chebyshev')
        #store points and weights
        self.points, self.weights = self.points_weights(n)


    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        if self.polytype == 'chebyshev':
            #evaluate the reoccurance relation for chebyshev
            b = np.array([1/4 for k in range(n-1)])
            b[0] *= 2
            u = np.pi

        elif self.polytype == 'legendre':
            #evaluate the reoccurance relation for legendre
            b = np.array([(k+1)**2/(4*(k+1)**2-1) for k in range(n-1)])
            u = 2

        #create jacobi matrix
        jacobi = diags([np.sqrt(b),np.sqrt(b)],[-1,1])
        jacobi = jacobi.toarray()
        #evaluate matrix for eigenvalues, eigenvectors
        evals, evecs = eig(jacobi)
        points = evals.real
        evecs = (evecs[0,:].real)**2
        weights = evecs*u
        #return eigenvalues as points and adjusted eigenvector first entries as weights
        return points, weights

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        #weight each solved point
        g = np.array([f(self.points[i])*self.inv_w(self.points[i]) for i in range(len(self.points))])
        #add them all together
        return sum(self.weights*g)


    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        #scale the coordinates
        h = lambda x: f((b-a)/2*x + (a+b)/2)
        #integrate over scaled area
        return (b-a)/2*self.basic(h)

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        #scale the coordinates
        h = lambda x,y: f((b1-a1)/2*x + (b1+a1)/2,(b2-a2)/2*y + (a2+b2)/2)
        #make all combinations of weights and solved points needed
        g = np.array([[h(self.points[i],self.points[j])*self.inv_w(self.points[j])*self.inv_w(self.points[i]) for j in range(len(self.points))] for i in range(len(self.points))])
        #sum all combinations and scale result
        return (b1-a1)*(b2-a2)/4*sum([sum([g[i,j]*self.weights[i]*self.weights[j] for j in range(len(self.points))]) for i in range(len(self.points))])


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    #using scipy calculate "exact" value
    the_value = norm.cdf(2) - norm.cdf(-3)
    #initialize variables needed for tests
    f = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)
    n = 5*np.arange(1,11)
    l_err = []
    c_err = []
    #ruu tests for various n's
    for i in n:
        #legendre
        leg = GaussianQuadrature(i)
        lvalue = leg.integrate(f,-3,2)
        l_err.append(abs(the_value-lvalue))
        #chebyshev
        cheb = GaussianQuadrature(i,polytype='chebyshev')
        cvalue = cheb.integrate(f,-3,2)
        c_err.append(abs(the_value-cvalue))
    #scipy results
    sp = np.ones_like(n)*abs(the_value-quad(f,-3,2)[0])
    #plot the results
    plt.plot(n,l_err,label='Legendre')
    plt.plot(n,c_err,label='Chebyshev')
    plt.plot(n,sp,label='Scipy')
    plt.semilogy()
    #make graph readable
    plt.xlabel('n')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Error Comparison in Gaussian Quadrature')
    plt.show()

