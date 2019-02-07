# solutions.py
#Name: Natalie Larsen
#Date: 11-20-2018
"""Volume 1: The SVD and Image Compression. Solutions File."""

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from imageio import imread


# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    eigs, vecs = la.eig(A.conj().T@A)
    svs = np.sqrt(eigs)
    #sort eigenvalues and eigenvectors accordingly
    sorter = list(zip(svs,vecs.T))
    sorter.sort(reverse=True, key=lambda tup: tup[0])
    svs = [x[0] for x in sorter]
    vecs = [x[1] for x in sorter]
    #find number of nonzero eigenvalues
    r_not = svs.count(0)
    r = len(svs) - r_not
    svs_1 = np.array(svs[:r])
    vecs_1 = np.array(vecs[:r])
    u_1 = (A@vecs_1)/svs_1

    return u_1, svs_1, vecs_1.conj().T


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    theta = np.linspace(0,2*np.pi,200)
    #Set S as unit circle
    S = np.array([np.cos(theta), np.sin(theta)])
    #Set E as orthogonal basis
    E = np.array([[1,0,0],[0,0,1]])
    U,Si,Vh = la.svd(A)
    Si = np.diag(Si)

    #plot original S and E
    first = plt.subplot(221)
    first.plot(S[0], S[1])
    first.plot(E[0], E[1])
    first.axis("equal")

    #rotate S,E and plot S,E
    second = plt.subplot(222)
    vhs = Vh@S
    vhe = Vh@E
    second.plot(vhs[0], vhs[1])
    second.plot(vhe[0], vhe[1])
    second.axis("equal")

    #scale S,E and plot S,E
    third = plt.subplot(223)
    sivhs = Si@vhs
    sivhe = Si@vhe
    third.plot(sivhs[0],sivhs[1])
    third.plot(sivhe[0],sivhe[1])
    third.axis([-4,4,-4,4])

    #rotate S,E and plot S,E
    fourth = plt.subplot(224)
    usivhs = U@sivhs
    usivhe = U@sivhe
    fourth.plot(usivhs[0],usivhs[1])
    fourth.plot(usivhe[0],usivhe[1])
    fourth.axis([-4,4,-4,4])

    plt.show()



# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, Si, Vh = la.svd(A)
    zeros = list(Si).count(0)
    #raise error if there are not enough nonzero singular values
    if len(Si) - zeros < s:
        raise ValueError("s > rank(A)")
    #Only save first s singular values for Si
    Si_hat = np.diag(Si[:s])
    #Save first s columns of U
    U_hat = U[:,:s]
    #Save first s rows of Vh
    Vh_hat = Vh[:s,:]

    # return new A and num of entries needed
    return U_hat@Si_hat@Vh_hat, U_hat.size+s+Vh_hat.size


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, Si, Vh = la.svd(A)
    for i in range(len(Si)):
        #raise error if there are no nonzero singular values less than error
        if Si[i] == 0:
            raise ValueError("A cannot be approximated in tolerance by matrix of lesser rank")

        #for greatest singular value less than error
        if Si[i] < err:
            #svd approximate for rank s
            Si_hat = np.diag(Si[:i])
            U_hat = U[:, :i]
            Vh_hat = Vh[:i, :]
            #return new A and num of entries needed
            return U_hat @ Si_hat @ Vh_hat, U_hat.size + i + Vh_hat.size




# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename) / 255
    size = image.shape
    orig_entries = image.size
    #colored
    if len(size) == 3:
        #plot original
        orig = plt.subplot(121)
        orig.imshow(image)
        orig.axis("off")
        #red in image
        R = image[:,:,0]
        #green in image
        G = image[:,:,1]
        #blue in image
        B = image[:,:,2]
        #approximate red, green and blue in range
        new_R, entries_R = svd_approx(R,s)
        new_R = np.clip(new_R,0,1)
        new_G, entries_G = svd_approx(G,s)
        new_G = np.clip(new_G,0,1)
        new_B, entries_B = svd_approx(B,s)
        new_B = np.clip(new_B,0,1)
        #stack all in one array
        new_image = np.dstack((new_R,new_G,new_B))
        #plot image
        new = plt.subplot(122)
        new.imshow(new_image)
        new.axis("off")
        #title image with saved number of entries
        plt.suptitle(str(orig_entries - (entries_R+entries_G+entries_B)) + " Entries")


    #grayscale
    else:
        #plot original
        orig = plt.subplot(121)
        orig.imshow(image, cmap="gray")
        orig.axis("off")
        #approximate the image
        new_A, entries = svd_approx(image,s)
        #plot it
        new = plt.subplot(122)
        new.imshow(new_A, cmap="gray")
        new.axis("off")
        #title image with saved number of entries
        plt.suptitle(str(orig_entries - entries) + " Entries")

    plt.show()

