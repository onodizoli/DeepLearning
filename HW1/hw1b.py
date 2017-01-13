from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import numpy.matlib
#%matplotlib inline
from matplotlib.pyplot import imshow

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

    imname: string
        name of file where image will be saved.
    '''
    g, axarr = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            # reshaping eigenvector to picture
            d = D[:,i*4+j].reshape(sz,sz)
            # plot picture
            axarr[i,j].imshow(d, cmap=cm.Greys_r)
    g.savefig(imname)
    plt.close(g)
    
def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''
    # reconstruct image
    X_hat = np.dot(D,c)
    # reshape image and add back mean
    image_new = X_hat.reshape(256,256) + X_mn
    # plot image
    ax.imshow(image_new, cmap=cm.Greys_r)  

if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''
    # Read filenames and sort them
    fileNames = os.listdir("./jaffe")
    fileNames.sort()
    
    # Load images
    I = np.zeros((len(fileNames), 256*256))
    ims = np.zeros((256, 256))    
    for i in range(0, len(fileNames)):
        ims[:,:] = Image.open("jaffe/" + fileNames[i])
        I[i,:] = ims.reshape((1,256*256))
    
    
    Ims = I.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    
    # Define parameters for gradient descent
    t_max = 35 # Max number of iteration
    eta = 0.001 # Parameter to set step size
    eps = 0.01 # Parameter to check convergence
    N = 16  # Number of eigenectors calculated
    d_j = np.zeros((256*256,N)) # Initializing eigenvectors (D matrix)
    Lam = np.zeros((N,1)) # Initialize Lambda vector (eigrnvalues)
    
    # Defining theano matricies
    X_T = T.fmatrix('X_T')
    di_T = T.fmatrix('di_T')
    dj_T = T.fmatrix('dj_T')
    lam_T = T.fmatrix('lam_T')
    
    # Iterating for every eigenvector
    for i in range (N):
        cond = True
        t=1
        d_i = np.random.random((256*256,1)) # Picking a startingpoint        
        # Calculating principle component via gradient descent
        while t<t_max and cond:
            # Function for calculating the gadient, I did the derivation instead using T.grad()
            s = -2*T.dot(X_T.T,T.dot(X_T,di_T)) + 2*T.dot(dj_T,(T.dot(dj_T.T,di_T))*lam_T)
            grad = theano.function([X_T, di_T, dj_T, lam_T], s, allow_input_downcast=True)
            # Calculating new di and normalize
            y = d_i - eta*grad(X, d_i, d_j[:,0:i], Lam[0:i,:])
            y = y/np.linalg.norm(y)
            # Chek if stupping condition is true (new di and old are close)
            if (np.linalg.norm(y-d_i,2))<eps:
                cond = False
            # update di
            d_i = y 
            t = t+1
        # Calculate eigenvalue
        lamda_maker = theano.function([X_T, di_T], T.dot(di_T.T,T.dot(X_T.T,T.dot(X_T,di_T))), allow_input_downcast=True)
        Lam[i,:] = lamda_maker(X,d_i)
        d_j[:,i:(i+1)]=d_i
        print str(i+1) + '. eigenvector found in ' + str(t) + ' steps.'
    # Assign eigenvectors to D matrix
    D = d_j
    
    c = np.dot(D.T, X.T)
    
    for i in range(0,200,10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')
    print 'Images created and saved to output.'
