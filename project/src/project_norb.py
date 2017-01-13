import numpy
#import scipy.io

import theano
import theano.tensor as T

from project_utils import load_norb
from project_nn import DropConnectMLP, train_nn, LeNetConvPoolLayer, LocallyConnectedLayer
from theano.tensor.signal import downsample

def test_dropout_model2(learning_rate_zero=0.001, L1_reg=0.00, L2_reg=0.00, n_epochs=[100,40,10],
             batch_size=128, n_hidden=512, n_hiddenLayers=1,
             verbose=False, activation = 'relu', momentum=1):
    """
    Wrapper function for training and testing MLP

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """

    # load the dataset; download the dataset if it is not present
    theano_shared=True
    datasets = load_norb(theano_shared=theano_shared)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    # compute number of minibatches for training, validation and testing
    if theano_shared:
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    else:
        n_train_batches = train_set_x.shape[0] // batch_size
        n_valid_batches = valid_set_x.shape[0] // batch_size
        n_test_batches = test_set_x.shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(1990))

    # Define kernel parameters
    nkerns=[64, 64, 64, 32]

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 54, 54))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 54, 54),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(3,3),
        pool_stride=(2,2)
    )

    # Zero padding
    zeros = numpy.zeros((batch_size, nkerns[0], 54, 54), dtype=theano.config.floatX)
    layer1_input = theano.shared(zeros)
    layer1_input = T.set_subtensor(layer1_input[:, :, 15:-15, 15:-15], layer0.output)

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer1_input,
        image_shape=(batch_size, nkerns[0], 54, 54),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(3,3),
        pool_stride=(2,2)
    )

    # Zero padding
    zeros = numpy.zeros((batch_size, nkerns[1], 54, 54), dtype=theano.config.floatX)
    layer2_input = theano.shared(zeros)
    layer2_input = T.set_subtensor(layer2_input[:, :, 15:-15, 15:-15], layer1.output)

    layer2 = LocallyConnectedLayer(
        rng,
        input=layer2_input,
        image_shape=(batch_size, nkerns[1], 54, 54),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        filter_stride=(1,1)
    )
    
    # Zero padding
    zeros = numpy.zeros((batch_size, nkerns[2], 54, 54), dtype=theano.config.floatX)
    layer3_input = theano.shared(zeros)
    layer3_input = T.set_subtensor(layer3_input[:, :, 1:-1, 1:-1], layer2.output)
    
    layer3 = LocallyConnectedLayer(
        rng,
        input=layer3_input,
        image_shape=(batch_size, nkerns[2], 54, 54),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        filter_stride=(1,1)
    )

    classifier_input = layer3.output.flatten(2)

    classifier = DropConnectMLP(
        rng=rng,
        srng=srng,
        input=classifier_input,
        n_in=nkerns[3]*52*52,
        n_hidden=n_hidden,
        n_out=6,
        n_hiddenLayers=n_hiddenLayers,
        activation=activation
    )
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # Add params from conv layers
    params = layer0.params + layer1.params + layer2.params + layer3.params + classifier.params

    # For rate decay
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](learning_rate_zero) )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in params]


    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    if momentum <1:
        updates = []
        # Updating parameters based on current gradient and precious step
        for param in params:
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - learning_rate*param_update))
            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))

    else:
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size] 
        }
    )
    # SET INFERENCE
    classifier.set_inference(inference=True)

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, learning_rate, learning_rate_zero, verbose)
        
