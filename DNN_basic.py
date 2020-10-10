# Generic L-layer 'straight in Python' Deep Neural Network implementation using basic Python/numpy.
# Mini-batches, regularization, optimization and batch normalization to be implemented.
#
# usage example:    params = L_layer_model(trainX, trainY, LAYERS, num_iterations=1500)
#                   predictTrain = predict(trainX, params, activations, trainY)
#                   predictDev = predict(devX, params, activations, devY)
#                   predictTest = predict(testX, params, activations, testY)

# main package
import numpy as np

# currently implemented activation functions
ACTIVATION_FUNCTIONS = ['sigmoid', 'relu', 'lrelu', 'tanh', 'softmax']

# seed global variable
SEED_VALUE = 3

# constant (Python dictionary) defining the model, layers dimensions and activation functions
# {'dims': [12288, 20, 7, 5, 1], 'activations': [None, 'relu', 'relu', 'relu', 'sigmoid']} is
# example of 4 layer model, 3 hidden layers with 20, 7, 5 units and relu activation function, and output layer
# (one unit for sigmoid function); 12288 is the # of # input features (not a layer of a network and doesn't need
# an activation function)
LAYERS = {'dims': [], 'activations': []}


def initialize_parameters(layers_dims):
    """
    Initialize weight matrices and bias vectors.
    
    Params:
    :layers_dims: python array (list) containing the dimensions (# of units)
                 of each layer in network
    
    Returns:
    :parameters: python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                bl -- bias vector of shape (layers_dims[l], 1)

    """
    np.random.seed(SEED_VALUE)
    L = len(layers_dims)  # number of layers in the network + input layer
    parameters = {}

    for lyr in range(1, L):
        parameters['W' + str(lyr)] = np.random.randn(layers_dims[lyr], layers_dims[lyr - 1]) * 0.01
        parameters['b' + str(lyr)] = np.zeros((layers_dims[lyr], 1))
        assert(parameters['W' + str(lyr)].shape == (layers_dims[lyr], layers_dims[lyr - 1]))
        assert(parameters['b' + str(lyr)].shape == (layers_dims[lyr], 1))
        
    return parameters


def linear_forward(A, W, b):
    """
    Linear part of a layer's forward propagation (wa + b), vectorized version.

    Parameters:
    :A: activations from previous layer (input data) of shape number of units of previous layer by number of examples
    :W: weights matrix, numpy array of shape size (# of units) of current layer by size of previous layer
    :b: bias vector, numpy array of shape size of the current layer by 1

    Returns:
    :Z: the input of the activation function, pre-activation parameter

    """
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    return Z


def sigmoid(Z):
    """
    Sigmoid activation function, vectorized version (array Z).
    
    Params:
    :Z: numpy array of any shape, output of the linear layer
    
    Returns:
    :A: post-activation output of sigmoid(z), same shape as Z

    """
    A = 1 / (1 + np.exp(-Z))

    assert(A.shape == Z.shape)

    return A


def relu(Z):
    """
    ReLU activation function, vectorized version (array Z).

    Params:
    :Z: numpy array of any shape, output of the linear layer

    Returns:
    :A: post-activation output of relu(Z), same shape as Z

    """
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)

    return A


def lrelu(Z, alpha=0.01):
    """
    Leaky ReLU activation function, vectorized version (array Z).

    Params:
    :Z: numpy array of any shape, output of the linear layer

    Returns:
    :A: post-activation output of lrelu(Z), same shape as Z

    """
    A = np.maximum(alpha * Z, Z)

    assert(A.shape == Z.shape)

    return A


def tanh(Z):
    """
    Tanh activation function, vectorized version (array Z).

    Params:
    :Z: numpy array of any shape, output of the linear layer

    Returns:
    :A: post-activation output of tanh(Z), same shape as Z

    """
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    assert(A.shape == Z.shape)

    return A


def softmax(Z):
    """
    Softmax activation function, vectorized version (array Z).

    Params:
    :Z: numpy array of any shape, output of the linear layer

    Returns:
    :A: post-activation output of softmax(Z), same shape as Z

    """
    Z_exp = np.exp(Z - np.max(Z))
    A = Z_exp / np.sum(Z_exp, axis=0)

    assert(A.shape == Z.shape)

    return A


def linear_activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for the LINEAR->ACTIVATION layer.

    Params:
    :A_prev: activations from previous layer (or input data for the first layer) of shape size of previous layer by
            number of examples
    :W: weights matrix, numpy array of shape size of current layer by size of previous layer
    :b: bias vector, numpy array of shape size of the current layer by 1
    :activation: the activation function to be used in layer, stored as a text string

    Returns:
    :A: the output of the activation function, post-activation value 
    :(linear_cache, Z): tuple containing linear cache and pre-activation parameter to be stored for computing the
                        backward pass efficiently

    """
    A, linear_cache, activation_cache = None, None, None

    # non-implemented activation function
    assert (activation in ACTIVATION_FUNCTIONS)

    Z = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "lrelu":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)
    elif activation == "softmax":
        A = softmax(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    return A, ((A_prev, W, b), Z)


def L_model_forward(X, parameters, activations):
    """
    Forward propagation for the layers in a network.
    
    Params:
    :X: data, numpy array of shape input size (# of features) by number of examples
    :parameters: output of initialize_parameters()
    :activations: list of activation functions for layers

    Returns:
    :A: last post-activation value (from the output layer, prediction probability)
    :caches: list of caches containing every cache of linear_activation_forward() (L-1 of them, indexed from 0 to L-1)

    """
    caches = []
    L = len(parameters)  # number of layers in the network
    A = X

    # forward propagation for L layers and add "cache" to the "caches" list
    for l in range(1, L+1):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activations[l])
        caches.append(cache)

    assert(A.shape == (1, X.shape[1]))
    
    return A, caches


def compute_cost(AL, Y, activation):
    """
    Calculates the cost defined by equation -[y*log(y_hat)+(1-y)*log(1-y_hat)], so presuming output layer activation
    function is sigmoid function.

    Params:
    :AL: probability vector corresponding to "label" predictions, y_hat in the aforementioned function, shape 1 by
        number of examples
    :Y: true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape 1 by number of examples

    Returns:
    :cost: cross-entropy cost
    
    """
    cost = None
    m = Y.shape[1]

    # only sigmoid or softmax
    assert (activation in [f for f in ACTIVATION_FUNCTIONS if f in ['sigmoid', 'softmax']])

    # compute loss from AL and Y
    if activation == 'sigmoid':
        cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))) / m
    elif activation == 'softmax':
        cost = - np.sum(np.multiply(Y, np.log(AL))) / m

    cost = np.squeeze(cost)  # to make sure cost's shape is as expected
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    """
    Linear portion of backward propagation for a single layer (layer l).

    Params:
    :dZ: gradient of the cost with respect to the linear output (of current layer l)
    :cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    :dA_prev: gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :dW: gradient of the cost with respect to W (current layer l), same shape as W
    :db: gradient of the cost with respect to b (current layer l), same shape as b
    
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def sigmoid_backward(dA, Z):
    """
    Backward propagation for SIGMOID activation function, vectorized version.

    Params:
    :dA: post-activation gradient, numpy array of any shape
    :Z: pre-activation parameter stored in cache during forward propagation

    Returns:
    :dZ: gradient of the cost function with respect to Z
    
    """
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def relu_backward(dA, Z):
    """
    Backward propagation for ReLU activation function, vectorized version.

    Params:
    :dA: post-activation gradient, numpy array of any shape
    :Z: pre-activation parameter stored in cache during forward propagation

    Returns:
    :dZ: gradient of the cost function with respect to Z
    
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def lrelu_backward(dA, Z, alpha=0.01):
    """
    Backward propagation for Leaky ReLU activation function, vectorized version.

    Params:
    :dA: post-activation gradient, numpy array of any shape
    :Z: pre-activation parameter stored in cache during forward propagation

    Returns:
    :dZ: gradient of the cost function with respect to Z

    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = alpha

    assert (dZ.shape == Z.shape)

    return dZ


def tanh_backward(dA, Z):
    """
    Backward propagation for tanh activation function, vectorized version.

    Params:
    :dA: post-activation gradient, numpy array of any shape
    :Z: pre-activation parameter stored in cache during forward propagation

    Returns:
    :dZ: gradient of the cost function with respect to Z

    """
    dZ = dA * (1 - np.power(tanh(Z), 2))
    assert (dZ.shape == Z.shape)

    return dZ


def linear_activation_backward(dA, cache, activation):
    """
    Backward propagation for the LINEAR->ACTIVATION layer.
    
    Parameters:
    :dA: post-activation gradient for current layer l 
    :cache: tuple of values (linear_cache, activation_cache) stored for computing backward propagation efficiently
    :activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    :dA_prev: gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :dW: gradient of the cost with respect to W (current layer l), same shape as W
    :db: gradient of the cost with respect to b (current layer l), same shape as b
    
    """
    dZ, dA_prev, dW, db = None, None, None, None

    # non-implemented activation function (softmax has a different algorithm, usually used only in output layer)
    # is different)
    assert (activation in [f for f in ACTIVATION_FUNCTIONS if f != 'softmax'])

    linear_cache, Z = cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif activation == "relu":
        dZ = relu_backward(dA, Z)
    elif activation == "lrelu":
        dZ = lrelu_backward(dA, Z)
    elif activation == 'tanh':
        dZ = tanh_backward(dA, Z)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, activations):
    """
    Backward propagation for the model.
    
    Params:
    :AL: probability vector, output of the forward propagation L_model_forward()
    :Y: true "label" vector (for example containing 0 if non-cat, 1 if cat)
    :caches: list of caches (linear and activation)
    :activations: list of activation functions for the model

    Returns:
    :grads: a dictionary with the gradients

    """
    grads = {}
    Y = Y.reshape(AL.shape)
    L = len(caches)  # the number of layers in the network

    # initializing the backpropagation (output layer must be either sigmoid or softmax)
    assert (activations[L] in [f for f in ACTIVATION_FUNCTIONS if f in ['sigmoid', 'softmax']])
    if activations[L] == 'sigmoid':
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
            = linear_activation_backward(dAL, caches[L - 1], activations[L])
    elif activations[L] == 'softmax':
        dZL = AL - Y
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
            = linear_backward(dZL, caches[L-1][0])

    # loop from l=L-1 to l=1
    for lyr in reversed(range(1, L)):
        grads["dA" + str(lyr-1)], grads["dW" + str(lyr)], grads["db" + str(lyr)] \
            = linear_activation_backward(grads["dA" + str(lyr)], caches[lyr-1], activations[lyr])

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent.
    
    Params:
    :parameters: Python dictionary containing weights and bias parameters 
    :grads: Python dictionary containing gradients, output of L_model_backward
    :learning_rate: model's learning rate
    
    Returns:
    :parameters: dictionary containing updated parameters

    """
    L = len(parameters)  # number of layers in the neural network

    for lyr in range(1, L+1):
        parameters["W" + str(lyr)] = parameters["W" + str(lyr)] - learning_rate * grads["dW" + str(lyr)]
        parameters["b" + str(lyr)] = parameters["b" + str(lyr)] - learning_rate * grads["db" + str(lyr)]
    
    return parameters


def L_layer_model(X, Y, layers, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    A L-layer neural network.

    Params:
    :X: data, numpy array of shape n_x (number of features) by m (number of examples)
    :Y: true "label" vector (for example containing 0 if cat, 1 if non-cat) of shape 1 by number of examples
    :layers_dims: list containing the input size and each layer size, of length number of layers + 1
    :learning_rate: learning rate of the gradient descent update rule
    :num_iterations: number of iterations of the optimization loop
    :print_cost: if True, it prints the cost at every 100 steps

    Returns:
    :parameters: parameters learnt by the model (used to predict)

    """

    # model layers not defined properly
    assert (len(layers['dims']) == len(layers['activations']))

    np.random.seed(SEED_VALUE)

    costs = []  # keep track of cost

    # parameters initialization
    parameters = initialize_parameters(layers['dims'])

    # loop gradient descent
    for i in range(0, num_iterations):
        # forward propagation
        AL, caches = L_model_forward(X, parameters, layers['activations'])
        # compute cost
        cost = compute_cost(AL, Y, layers['activations'][-1])
        # backward propagation
        grads = L_model_backward(AL, Y, caches, layers['activations'])
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, float(cost)))
            costs.append(cost)

    return parameters


def predict(X, parameters, activations, Y=None):
    """
    Function used to predict the results of a L-layer neural network.

    Params:
    :X: data set of examples to label, numpy array of shape n_x (number of features) by m (number of examples)
    :parameters: parameters of the trained model, returned by L_layer_model()
    :activations: list of activation functions for the model
    :Y: if given, true "label" vector of shape 1 by number of examples to print the accuracy

    Returns:
    :P: predictions for the given dataset X, shape 1 by number of examples

    """
    m = X.shape[1]
    P = np.zeros((1, m))

    # forward propagation
    probabilities, _ = L_model_forward(X, parameters, activations)

    # convert probabilities to 0/1 predictions
    # currently for sigmoid in output layer only
    assert (activations[-1] == 'sigmoid')
    for i in range(0, probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            P[0, i] = 1
        else:
            P[0, i] = 0

    if Y is not None:
        print("Accuracy: " + str(np.sum((P == Y) / m)))

    return P
