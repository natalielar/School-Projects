# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name> Natalie Larsen
<Class> 001
<Date> 10-30-18
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la
import functools
from cmath import sqrt


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    #QR reduce A
    Q,R = la.qr(A,mode="economic")
    b2 = Q.T @ b
    #backsolve for x
    x = la.solve_triangular(R,b2)
    return x



# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    data = np.load("housing.npy")
    #b is column vector of prices
    b = np.vstack(data[:,1])
    #A is column of year and column of 1
    a = data[:,0]
    x = a.shape
    o = np.ones(x)
    A = np.column_stack((a,o))
    #Solve least squares
    sol = least_squares(A,b)
    domain = np.linspace(0,17,35)
    lin = sol[0] * domain + sol[1]

    #plot data and line
    plt.plot(data[:,0],data[:,1],"*")
    plt.plot(domain, lin)
    plt.show()




# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    data = np.load("housing.npy")
    #set housing prices as vector b
    b = np.vstack(data[:,1])
    a = data[:,0]
    #create Vandermonde matrices deg 3,6,9,12
    A_3 = np.vander(a, 3)
    A_6 = np.vander(a, 6)
    A_9 = np.vander(a, 9)
    A_12 = np.vander(a, 12)
    #find least squares solution deg 3,6,9,12
    sol_3 = la.lstsq(A_3,b)[0]
    sol_6 = la.lstsq(A_6,b)[0]
    sol_9 = la.lstsq(A_9,b)[0]
    sol_12 = la.lstsq(A_12,b)[0]
    domain = np.linspace(0,16,50)
    #orgainze data horizontally
    f_3 = np.poly1d(np.hstack(sol_3))
    f_6 = np.poly1d(np.hstack(sol_6))
    f_9 = np.poly1d(np.hstack(sol_9))
    f_12 = np.poly1d(np.hstack(sol_12))

    #plot each polynomial in a graph w/ data points
    plt3 = plt.subplot(221)
    plt3.plot(data[:, 0], data[:, 1], "*")
    plt3.plot(domain, f_3(domain))
    plt6 = plt.subplot(222)
    plt6.plot(data[:, 0], data[:, 1], "*")
    plt6.plot(domain, f_6(domain))
    plt9 = plt.subplot(223)
    plt9.plot(data[:, 0], data[:, 1], "*")
    plt9.plot(domain, f_9(domain))
    plt12 = plt.subplot(224)
    plt12.plot(data[:, 0], data[:, 1], "*")
    plt12.plot(domain, f_12(domain))

    plt.show()



def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    data = np.load("ellipse.npy")
    xk = np.array(data[:,0])
    yk = np.array(data[:,1])
    #create least squares set up:
    #[ax^2 bx cxy dy ey^2 = 1]
    A = np.array([xk**2,xk,xk*yk,yk,yk**2]).T
    b = np.ones_like(xk)
    #solve least squares
    sol = la.lstsq(A,b)[0]
    a,b,c,d,e = sol[0],sol[1],sol[2],sol[3],sol[4]

    #plot ellipse and points
    plot_ellipse(a,b,c,d,e)
    plt.plot(xk,yk,'*')

    plt.show()





# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = A.shape
    #create random vector of length n
    x = np.random.rand(n)
    #normalize x
    x = x/la.norm(x)
    for i in range(N):
        #form xk and normalize it
        x_new = A@x
        x_new = x_new / la.norm(x_new)
        #if change in x is smaller than tol, break
        if la.norm(x_new - x) <= tol:
            x = x_new
            break
        x = x_new
    return x.T@A@x, x


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m,n = A.shape
    #Put A in upper Hessenberg form
    S = la.hessenberg(A)
    for k in range(N):
        #get QR decomp of Ak
        Q,R = la.qr(S)
        #recombine Rk and Qk into Ak+1
        S = R@Q
    #initialize empty list of eigenvalues
    eigs = []
    i = 0
    while i < n:
        #if Si is 1x1
        if i == n-1 or abs(S[i+1][i]) < tol:
            eigs.append(S[i][i])
        #if Si is 2x2
        else:
            b = S[i][i] + S[i+1][i+1]
            c = S[i][i]*S[i+1][i+1] + S[i][i+1]*S[i+1][i]
            #find eigenvalues w/ quadratic formula
            first = (b + sqrt(b**2 - 4*c))/2
            second = (b - sqrt(b**2 - 4*c))/2
            eigs.append(first)
            eigs.append(second)
            i += 1
        #move to next Si
        i += 1

    return eigs


