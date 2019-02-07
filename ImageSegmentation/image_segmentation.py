# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name> Natalie Larsen
<Class> 001
<Date> 11-6-18
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph,linalg
import scipy.linalg as la
from imageio import imread
from matplotlib import pyplot as plt



# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    #initialize A shape matrix
    L = np.zeros_like(A)
    m,n = L.shape
    #Make L the diagonal matrix
    for i in range(n):
        d = sum(A[i,:])
        L[i,i] = d
    #L = D-A
    return L-A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    #Find the eigenvalues of L
    eigs = la.eig(L)[0]
    zeros = 0
    eigs.sort()
    #count the zero eigenvalue for components
    for e in eigs:
        if abs(np.real(e)) < tol:
            zeros += 1
    #return second smallest eigenvalue as connectivity
    return zeros, np.real(eigs[1])

# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)
        #scale the image and store ita s attribute
        self.scaled = image/255
        size = image.shape
        #colored image
        if len(size) == 3:
            #lower to dimension 2 and initialize brightness attribute
            self.brightness = self.scaled.mean(axis=2)
            self.brightness = np.ravel(self.brightness)
        #gray image
        else:
            #initialize brightness attribute
            self.brightness = np.ravel(self.scaled)


    # Problem 3
    def show_original(self):
        """Display the original image."""
        #for gray scale images
        if len(self.scaled.shape) == 2:
            plt.imshow(self.scaled, cmap="gray")
            plt.axis("off")
        #for colored images
        else:
            plt.imshow(self.scaled)
            plt.axis("off")
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        if len(self.scaled.shape) == 2:
            m,n = self.scaled.shape
        else:
            m,n,o = self.scaled.shape
        #initialize adjacency matrix
        A = sp.lil_matrix((m*n,m*n))
        D = np.zeros((m*n,1))
        for i in range(m*n):
            #find adjacent neighbors and their distance
            J,W = get_neighbors(i,r,m,n)
            weights = []
            w = 0
            for j in J:
                #calculate the weight for each neighbor
                new_one = np.exp(-abs(self.brightness[i]-self.brightness[j])/
                                 sigma_B2 - W[w]/sigma_X2)
                weights.append(new_one)
                w += 1
            #save weights in position in adjacency matrix
            A[i,J] = weights
            D[i] = sum(W)
        A = sp.csc_matrix(A)
        return A,D


    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        if len(self.scaled.shape) == 2:
            m,n = self.scaled.shape
        else:
            m,n,o = self.scaled.shape
        #find laplacian matrix of adjacency matrix
        L = sp.csgraph.laplacian(A)
        #find D^-1/2
        D1_2 = np.hstack(D**(-1/2))
        Dm = sp.diags(D1_2,0)
        Q = Dm@L@Dm
        #find second smallest eigenvector of D^-1/2LD^-1/2 and reshape like image
        values, vectors = sp.linalg.eigsh(Q,which="SM",k=2)
        m = vectors[:,1].reshape((m,n))
        #create mask; positive values true, nonpositive false
        mask = m > 0
        return mask


    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        #find adjacency matrix
        A,D = self.adjacency(r,sigma_B,sigma_X)
        #find mask
        mask = self.cut(A,D)
        mask_inv = ~mask
        #original image
        orig = plt.subplot(131)
        #positive image
        pos = plt.subplot(132)
        #negative image
        neg = plt.subplot(133)
        #if gray scale, show the 3 images
        if len(self.scaled.shape) == 2:
            orig.imshow(self.scaled, cmap="gray")
            orig.axis("off")
            pos.imshow(mask*self.scaled,cmap="gray")
            pos.axis("off")
            neg.imshow(mask_inv*self.scaled,cmap="gray")
            neg.axis("off")
        #if colored show the 3 images
        else:
            orig.imshow(self.scaled)
            orig.axis("off")
            new_mask = np.dstack((mask,mask,mask))
            pos.imshow(new_mask*self.scaled)
            pos.axis("off")
            neg.imshow((~new_mask)*self.scaled)
            neg.axis("off")
        plt.show()


# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
