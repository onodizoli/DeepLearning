"""
Source Code for Homework 3 of ECBM E6040, Spring 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
"""
import os
import sys
import numpy
import scipy.io

import theano
import theano.tensor as T

import cPickle
from struct import unpack
import gzip
from numpy import zeros, uint8
import tarfile
import bz2

import theano
from theano import tensor as T
from theano.tensor.signal import downsample
from cont_norm import crop3, contranorma

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_svhn(ds_rate=None, theano_shared=True):

    # Download the SVHN dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'http://ufldl.stanford.edu/housenumbers/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    train_dataset = check_dataset('train_32x32.mat')
    test_dataset = check_dataset('test_32x32.mat')

    # Load the dataset
    train_set = scipy.io.loadmat(train_dataset)
    test_set = scipy.io.loadmat(test_dataset)

    
    def convert_data_format(data):
        '''
            X = numpy.reshape(data['X'],
                        (numpy.prod(data['X'].shape[:-1]), data['X'].shape[-1]),
                        order='C').T / 255.
        '''
        X = data['X'].transpose(2,0,1,3)
        X = X.reshape((numpy.prod(X.shape[:-1]), X.shape[-1]),order='C').T #/255.
        print numpy.shape(X)
        X = numpy.apply_along_axis(contranorma,1,X)
        X = numpy.apply_along_axis(crop3,1,X)
        
        y = data['y'].flatten()
        y[y == 10] = 0
        return (X,y)
    train_set = convert_data_format(train_set)
    test_set = convert_data_format(test_set)

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval



def load_mnist(croped=False, theano_shared=True):
    ''' Loads the MNIST dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''


    # Download the SVHN dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            "MNIST",
            dataset
        )
        if (not os.path.isfile(new_path)):
            print('Downloading data')
            from six.moves import urllib
            urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                                       '../data/MNIST/train-images-idx3-ubyte.gz')
            urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                                       '../data/MNIST/train-labels-idx1-ubyte.gz')
            urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                                       '../data/MNIST/t10k-images-idx3-ubyte.gz')
            urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
                                       '../data/MNIST/t10k-labels-idx1-ubyte.gz')
            print('Downloading complete, pickeling data')
            x, y = get_labeled_data('../data/MNIST/train-images-idx3-ubyte.gz','../data/MNIST/train-labels-idx1-ubyte.gz')
            cPickle.dump(x, open('../data/MNIST/train.p', 'wb'))
            cPickle.dump(y, open('../data/MNIST/trainLabel.p', 'wb'))
            x2, y2 = get_labeled_data('../data/MNIST/t10k-images-idx3-ubyte.gz','../data/MNIST/t10k-labels-idx1-ubyte.gz')
            cPickle.dump(x2, open('../data/MNIST/test.p', 'wb'))
            cPickle.dump(y2, open('../data/MNIST/testLabel.p', 'wb'))
            print('Pickle complete')


    check_dataset('train.p')
    #test_dataset = check_dataset('test_32x32.mat')

    # Load the dataset
    train_set_x = cPickle.load(open('../data/MNIST/train.p', 'rb'))
    train_set_y = cPickle.load(open('../data/MNIST/trainLabel.p', 'rb'))
    test_set_x = cPickle.load(open('../data/MNIST/test.p', 'rb'))
    test_set_y = cPickle.load(open('../data/MNIST/testLabel.p', 'rb'))


    # Convert data format
    def convert_data_format(data_x, data_y, croped = False):
        X = data_x.reshape(len(data_x), data_x.shape[1]*data_x.shape[2])/255.0
        y = data_y.flatten()
        y[y == 10] = 0
        if croped == True:
            A = numpy.apply_along_axis(crop,1,X)
            B = numpy.apply_along_axis(crop,1,X)
            C = numpy.apply_along_axis(crop,1,X)
            D = numpy.apply_along_axis(crop,1,X)
            F = numpy.concatenate((A,B,C,D), axis=0)
            Y = numpy.concatenate((y,y,y,y), axis=0)
            return (F,Y)
        else:
            return (X,y)

    train_set = convert_data_format(train_set_x, train_set_y)
    test_set = convert_data_format(test_set_x,test_set_y)

    # Downsample the training dataset if specified

    train_set_len = len(train_set[1])
    '''
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]
    '''

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval

def load_cifar10(theano_shared=True):

    ''' Loads the CIFAR-10 dataset

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''

    # Path to data folder
    data_path = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data"
    )

    # Download dataset if it is not present
    def check_dataset(dataset, datapath):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            datapath,
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'https://www.cs.toronto.edu/~kriz/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    # Get path to downloaded .tar file
    datapath = check_dataset("cifar-10-python.tar.gz", data_path)

    # Unpack CIFAR-10 from tarball
    def unpack_dataset(dataset, datapath):
        # Check if dataset already unpacked
        if os.path.isdir(os.path.join(data_path, "cifar-10-batches-py")):
            print "Data already unpacked"
            return
        # If not already unpacked
        if dataset.endswith("tar.gz"):
            with tarfile.open(dataset,"r") as tar:
                tar.extractall(path=data_path)
                print "Unpacked Dataset"

    # .tar.gz -> extracted files
    unpack_dataset(datapath, data_path)

    # Unpickle extracted files
    def unpickle(pickled_file):
        with open(pickled_file,"rb") as pf:
            d = cPickle.load(pf)
        data, labels = d['data'], d['labels']

        return numpy.divide(data, 255.0), labels

    # Update path to data folder
    data_path = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data",
        "cifar-10-batches-py"
    )

    # Separate and concatenate data
    data_1, labels_1 = unpickle(os.path.join(data_path, "data_batch_1"))
    data_2, labels_2 = unpickle(os.path.join(data_path, "data_batch_2"))
    data_3, labels_3 = unpickle(os.path.join(data_path, "data_batch_3"))
    data_4, labels_4 = unpickle(os.path.join(data_path, "data_batch_4"))
    train_set_x, train_set_y = numpy.vstack([data_1, data_2, data_3, data_4]), \
                               numpy.hstack([labels_1, labels_2, labels_3, labels_4])
    train_set = (train_set_x, train_set_y)

    valid_set_x, valid_set_y = unpickle(os.path.join(data_path, "data_batch_5"))
    valid_set = (valid_set_x, valid_set_y)

    test_set_x, test_set_y = unpickle(os.path.join(data_path, "test_batch"))
    test_set = (test_set_x, test_set_y)

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval
    
def readNums(file_handle, num_type, count):
    """
    Reads 4 bytes from file, returns it as a 32-bit integer.
    """
    num_bytes = count * numpy.dtype(num_type).itemsize
    string = file_handle.read(num_bytes)
    return numpy.fromstring(string, dtype = num_type)
    
def readHeader(file_handle, debug=False, from_gzip=None):
    """
    :param file_handle: an open file handle. 
    :type file_handle: a file or gzip.GzipFile object

    :param from_gzip: bool or None
    :type from_gzip: if None determine the type of file handle.

    :returns: data type, element size, rank, shape, size
    """

    if from_gzip is None:
        from_gzip = isinstance(file_handle,
                              (gzip.GzipFile, bz2.BZ2File))

    key_to_type = { 0x1E3D4C51 : ('float32', 4),
                    # what is a packed matrix?
                    # 0x1E3D4C52 : ('packed matrix', 0),
                    0x1E3D4C53 : ('float64', 8),
                    0x1E3D4C54 : ('int32', 4),
                    0x1E3D4C55 : ('uint8', 1),
                    0x1E3D4C56 : ('int16', 2) }

    type_key = readNums(file_handle, 'int32', 1)[0]
    elem_type, elem_size = key_to_type[type_key]
    if debug: 
        print "header's type key, type, type size: ", \
            type_key, elem_type, elem_size
    if elem_type == 'packed matrix':
        raise NotImplementedError('packed matrix not supported')

    num_dims = readNums(file_handle, 'int32', 1)[0]
    if debug: 
        print '# of dimensions, according to header: ', num_dims

    if from_gzip:
        shape = readNums(file_handle, 
                         'int32',
                         max(num_dims, 3))[:num_dims]
    else:
        shape = numpy.fromfile(file_handle, 
                               dtype='int32',
                               count=max(num_dims, 3))[:num_dims]

    if debug: 
        print 'Tensor shape, as listed in header:', shape

    return elem_type, elem_size, shape
    
def parseNORBFile(file_handle, subtensor=None, debug=False):
    """
    Load all or part of file 'f' into a numpy ndarray
    :param file_handle: file from which to read file can be opended with
      open(), gzip.open() and bz2.BZ2File() @type f: file-like
      object. Can be a gzip open file.

    :param subtensor: If subtensor is not None, it should be like the
      argument to numpy.ndarray.__getitem__.  The following two
      expressions should return equivalent ndarray objects, but the one
      on the left may be faster and more memory efficient if the
      underlying file f is big.

       read(f, subtensor) <===> read(f)[*subtensor]

      Support for subtensors is currently spotty, so check the code to
      see if your particular type of subtensor is supported.
      """

    elem_type, elem_size, shape = readHeader(file_handle,debug)
    beginning = file_handle.tell()

    num_elems = numpy.prod(shape)

    result = None
    if isinstance(file_handle, (gzip.GzipFile, bz2.BZ2File)):
        assert subtensor is None, \
            "Subtensors on gzip files are not implemented."
        result = readNums(file_handle, 
                          elem_type, 
                          num_elems*elem_size).reshape(shape)
    elif subtensor is None:
        result = numpy.fromfile(file_handle,
                                dtype = elem_type,
                                count = num_elems).reshape(shape)
    elif isinstance(subtensor, slice):
        if subtensor.step not in (None, 1):
            raise NotImplementedError('slice with step', subtensor.step)
        if subtensor.start not in (None, 0):
            bytes_per_row = numpy.prod(shape[1:]) * elem_size
            file_handle.seek(beginning+subtensor.start * bytes_per_row)
        shape[0] = min(shape[0], subtensor.stop) - subtensor.start
        result = numpy.fromfile(file_handle, 
                                dtype=elem_type,
                                count=num_elems).reshape(shape)
    else:
        raise NotImplementedError('subtensor access not written yet:',
                                  subtensor) 
                
    return result
    
def load_norb(theano_shared=True):
    
    ''' Loads the NORB stereo image dataset

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''

    # Path to data folder
    data_path = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data"
    )

    # CIFAR-10 the SVHN dataset if it is not present
    def check_dataset(dataset, datapath):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            datapath,
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path
    
    # Unpack dataset from tarball
    def unpack_dataset(dataset, datapath):
        # Check if dataset already unpacked
        if os.path.isfile(dataset[:-3]):
            print "Data already unpacked"
            return dataset[:-3]
        # If not already unpacked
        import gzip
        if dataset.endswith(".gz"):
            with gzip.open(dataset,"rb") as g:
                data = g.read()
            with open(dataset[:-3],"w") as d:
                d.write(data)
            print "Unpacked Dataset"
            return dataset[:-3]
        print "Incompatible extension (must be .gz)"

    # .tar.gz -> extracted files    # Get path to downloaded .tar file
    datapath = check_dataset("norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat.gz", data_path)
    datapath = unpack_dataset(datapath, data_path)
    with open(datapath, 'r') as test_cat:
        test_set_y = parseNORBFile(test_cat)
        test_set_y = numpy.hstack([test_set_y, test_set_y]).tolist()
         
    datapath = check_dataset("norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat.gz", data_path)
    datapath = unpack_dataset(datapath, data_path)
    with open(datapath, 'r') as test_dat:
        test_set_x_L, test_set_x_R = numpy.split(parseNORBFile(test_dat), 2, axis=1)
        test_set_x_L, test_set_x_R = test_set_x_L.reshape(29160, 108, 108), test_set_x_R.reshape(29160, 108, 108)
        test_set_x = numpy.vstack([test_set_x_L, test_set_x_R])
        del test_set_x_L, test_set_x_R
    
    datapath = check_dataset("norb-5x46789x9x18x6x2x108x108-training-01-cat.mat.gz", data_path)
    datapath = unpack_dataset(datapath, data_path)
    with open(datapath, 'r') as train_cat:
        train_set_y = parseNORBFile(train_cat)
        train_set_y = numpy.hstack([train_set_y, train_set_y]).tolist()
    
    datapath = check_dataset("norb-5x46789x9x18x6x2x108x108-training-01-dat.mat.gz", data_path)
    datapath = unpack_dataset(datapath, data_path)
    with open(datapath, 'r') as train_dat:
        train_set_x_L, train_set_x_R = numpy.split(parseNORBFile(train_dat), 2, axis=1)
        train_set_x_L, train_set_x_R = train_set_x_L.reshape(29160, 108, 108), train_set_x_R.reshape(29160, 108, 108)
        train_set_x = numpy.vstack([train_set_x_L, train_set_x_R])
        del train_set_x_L, train_set_x_R
        
    # Rearrange data to make a train, validate and test set
    valid_set_x, test_set_x = test_set_x[:29160,:,:], test_set_x[29160:,:,:]
    valid_set_y, test_set_y = test_set_y[:29160], test_set_y[29160:]
        
    mat = T.fmatrix('mat')
    pooled_out = downsample.max_pool_2d(
        input=mat,
        ds=(2,2),
        ignore_border=True
    )
    ds = theano.function([mat], pooled_out)
    
    # Downsample
    train_set_x_temp = numpy.zeros((58320, 54, 54), dtype=theano.config.floatX)
    for i in xrange(58320):
        train_set_x_temp[i, :, :] = ds(train_set_x[i,:,:])
        
    test_set_x_temp = numpy.zeros((29160, 54, 54), dtype=theano.config.floatX)
    valid_set_x_temp = numpy.zeros((29160, 54, 54), dtype=theano.config.floatX)
    for i in xrange(29160):
        test_set_x_temp[i, :, :] = ds(test_set_x[i,:,:])
        valid_set_x_temp[i, :, :] = ds(valid_set_x[i,:,:])
    
    test_set = (numpy.divide(test_set_x_temp, 255.0).reshape((29160, 54*54)), test_set_y)
    valid_set = (numpy.divide(valid_set_x_temp, 255.0).reshape((29160, 54*54)), valid_set_y)
    train_set = (numpy.divide(train_set_x_temp, 255.0).reshape((58320, 54*54)), train_set_y)
        
    del test_set_x, test_set_y, valid_set_x, valid_set_y, train_set_x, train_set_y, \
        test_set_x_temp, train_set_x_temp, valid_set_x_temp
    
    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval

def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=uint8)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 10000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

def crop(A):
    length = 28
    newlength = 24
    channel=1
    x = numpy.random.randint(0,length-newlength+1)
    y = numpy.random.randint(0,length-newlength+1)
    B = numpy.zeros((1,newlength*newlength))
    for i in range (0,newlength):      
        B[0,i*newlength*channel:(i*newlength+newlength)*channel]=A[(y*length+i*length+x)*channel:(y*length+i*length+x+newlength)*channel]
    return B.reshape(newlength*newlength)


#load_norb()
#load_cifar10()