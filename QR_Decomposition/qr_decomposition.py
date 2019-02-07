# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name> Natalie Larsen
<Class> 001
<Date> 10-23-18
"""
import scipy.linalg as la
import numpy as np

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m,n = A.shape
    Q = A.copy().astype(float)
    #n x n matrix of zeros
    R = np.zeros((n,n))
    for i in range(0,n):
        R[i,i] = la.norm(Q[:,i])
        #normalize ith column of Q
        Q[:,i] = Q[:,i]/R[i,i]
        for j in range(i+1,n):
            R[i,j] = Q[:,j].T@Q[:,i]
            #orthogonalize the jth column of Q
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
    return Q,R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q,R = qr_gram_schmidt(A)
    #self explanatory....
    return abs(np.prod(np.diag(R)))


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    #compute Q and R
    Q,R = qr_gram_schmidt(A)
    #calculate y = Qtb
    y = Q.T@b
    m,n = R.shape
    #back substitution to solve Rx=y for x
    x = list([y[n-1]/R[n-1,n-1]])
    for f in range(1,n):
        x_new = y[n-f-1]
        #apply transformation to rest of the row
        for e in range(0,f):
            x_new -= R[n-f-1][n-e-1]*x[e]
        x_new = x_new / R[n-f-1][n-f-1]
        x.append(x_new)
        x = list(x[::])
    return np.array(x[::-1])

# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    s = lambda x: 1 if x >= 0 else -1

    m,n = A.shape
    R = A.copy().astype(float)
    #create m x m identity matrix
    Q = np.eye(m)
    for k in range(0,n):
        u = R[k:,k].copy().astype(float)
        #u[0] will be the first entry of u
        u[0] = u[0] + s(u[0]) * la.norm(u)
        #normalize u
        u = u/la.norm(u)
        #apply reflection to R
        R[k:,k:] = R[k:,k:] - np.outer(2*u,(u.T@R[k:,k:]))
        #apply reflection to Q
        Q[k:,:] = Q[k:,:] - np.outer(2*u,(u.T@Q[k:,:]))
    return Q.T,R


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    s = lambda x: 1 if x >= 0 else -1

    m,n = A.shape
    H = A.copy().astype(float)
    #m x m identity matrix
    Q = np.eye(m)
    for k in range(0,n-2):
        u = H[k+1:,k].copy().astype(float)
        u[0] = u[0] + s(u[0]) * la.norm(u)
        u = u/la.norm(u)
        #apply Qk to H
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,u.T@H[k+1:,k:])
        #apply Qtk to H
        H[:,k+1:] = H[:,k+1:] -2*np.outer((H[:,k+1:]@u),u.T)
        #Apply Qk to Q
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u,u.T@Q[k+1:,:])
    return H,Q.T
