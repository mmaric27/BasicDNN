# Generic L-layer 'straight in Python' Deep Neural Network implementation using basic Python/numpy.
#
#Input data is supposed to be stacked in a matrix of n_x by m, where n_x is a number of input features for an example
# and m is the number of training examples.
#Output layer can be either Sigmoid or Softmax classifier.
#Implemented activation functions: Sigmoid, ReLU, Leaky ReLU, Tanh, Softmax.
#Implemented weights initialization methods: zeros, random, He, Xavier.
#
# usage example:    model = L_layer_model(trainX, trainY, MODEL, num_iterations=1500)
#                   predictTrain = predict(trainX, model, trainY)
#                   predictDev = predict(devX, model, devY)
#                   predictTest = predict(testX, model, testY)

# main package
import numpy as np

# currently implemented activation functions
ACTIVATION_FUNCTIONS = ['sigmoid', 'relu', 'lrelu', 'tanh', 'softmax']
# currently implemented wights initializations
INITIALIZATIONS = ['zeros', 'random', 'xavier', 'he']

# seed global variable
SEED_VALUE = 0

# tuple defining the model (layers dimensions, activation functions and weights initialization method), for
# instance ((20, 'relu', 'he'), (7, 'relu', 'he'), (5, 'relu', 'he'), (1, 'sigmoid', 'random')) is an example of 4 layer
# model, 3 hidden layers with 20, 7, 5 units and relu activation function and he weights initialization, and output
# layer with one unit, sigmoid function and random initialization
MODEL = ()


def initialize_model(n_x, model):
    """
    Initialize model, its weight matrix, bias vector and activation functions.
    
    Args:
        n_x (integer): number of input features
        model (tuple): network layer definitions, each element in a tuple is a tuple containing number of units in a
                    layer, activation function and weights initialization method for a layer

    Returns:
        M (list): network layer definitions, each element in a list is a list containing weight and bias parameters, and
                activation function for a layer

    """
    M = []

    for layer_id in range(len(model)):
        # initialization method
        method = model[layer_id][2]
        # number of units in a layer
        n = model[layer_id][0]
        # number of units in a previous layer
        n_prev = model[layer_id-1][0] if layer_id > 0 else n_x

        if method == 'zeros':
            W = np.zeros((n, n_prev))
        elif method == 'xavier':
            W = np.random.randn(n, n_prev) * np.sqrt(1 / n_prev)
        elif method == 'he':
            W = np.random.randn(n, n_prev) * np.sqrt(2 / n_prev)
        else:
            W = np.random.randn(n, n_prev) * 0.01

        b = np.zeros((n, 1))

        M.append([W, b, model[layer_id][1]])

    return M


def linear_forward(A, W, b):
    """
    Linear part of a layer's forward propagation (wa + b), vectorized version.

    Args:
        A (ndarray): activations from previous layer (input data) of shape number of units of previous layer by number
                    of examples
        W (ndarray): weights matrix of shape size (# of units) of current layer by size of previous layer
        b (ndarray): bias vector, numpy array of shape size of the current layer by 1

    Returns:
        Z (ndarray): the input of the activation function, pre-activation parameter

    """
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    return Z


def sigmoid(Z):
    """
    Sigmoid activation function, vectorized version (array Z).
    
    Args:
        Z (ndarray): array of any shape, output of the linear layer
    
    Returns:
        A (ndarray): post-activation output of sigmoid(z), same shape as Z

    """
    A = 1 / (1 + np.exp(-Z))

    assert(A.shape == Z.shape)

    return A


def relu(Z):
    """
    ReLU activation function, vectorized version (array Z).

    Args:
        Z (ndarray): numpy array of any shape, output of the linear layer

    Returns:
        A (ndarray): post-activation output of relu(Z), same shape as Z

    """
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)

    return A


def lrelu(Z, alpha=0.01):
    """
    Leaky ReLU activation function, vectorized version (array Z).

    Args:
        Z (ndarray): numpy array of any shape, output of the linear layer

    Returns:
        A (ndarray): post-activation output of lrelu(Z), same shape as Z

    """
    A = np.maximum(alpha * Z, Z)

    assert(A.shape == Z.shape)

    return A


def tanh(Z):
    """
    Tanh activation function, vectorized version (array Z).

    Args:
        Z (ndarray): numpy array of any shape, output of the linear layer

    Returns:
        A (ndarray): post-activation output of tanh(Z), same shape as Z

    """
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    assert(A.shape == Z.shape)

    return A


def softmax(Z):
    """
    Softmax activation function, vectorized version (array Z).

    Args:
        Z (ndarray): numpy array of any shape, output of the linear layer

    Returns:
        A (ndarray): post-activation output of softmax(Z), same shape as Z

    """
    Z_exp = np.exp(Z - np.max(Z))
    A = Z_exp / np.sum(Z_exp, axis=0)

    assert(A.shape == Z.shape)

    return A


def linear_activation_forward(A_prev, W, b, g):
    """
    Forward propagation for the LINEAR->ACTIVATION layer.

    Args:
        A_prev (ndarray): activations from previous layer (or input data for the first layer) of shape size of previous
                        layer by number of examples
        W (ndarray): weights matrix, numpy array of shape size of current layer by size of previous layer
        b (ndarray): bias vector, numpy array of shape size of the current layer by 1
        g (string): activation function to be used in layer

    Returns:
        A (ndarray): the output of the activation function, post-activation value
        linear_cache, Z (tuple): tuple containing linear cache and pre-activation parameter to be stored for computing
                                the backward pass efficiently

    """
    A, linear_cache, activation_cache = None, None, None

    # non-implemented activation function
    assert (g in ACTIVATION_FUNCTIONS)

    Z = linear_forward(A_prev, W, b)

    if g == "sigmoid":
        A = sigmoid(Z)
    elif g == "relu":
        A = relu(Z)
    elif g == "lrelu":
        A = lrelu(Z)
    elif g == "tanh":
        A = tanh(Z)
    elif g == "softmax":
        A = softmax(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    return A, ((A_prev, W, b), Z)


def L_model_forward(X, M):
    """
    Forward propagation for the layers in a network.
    
    Args:
        X (ndarray): data, array of shape input size (# of features) by number of examples
        M (list): output of initialize_model()

    Returns:
        A (ndarray): last post-activation value (from the output layer, prediction probability)
        caches (list): list of caches containing every cache of linear_activation_forward()

    """
    caches = []
    L = len(M)  # number of layers in the network
    A = X

    # forward propagation for L layers and add "cache" to the "caches" list
    for layer in range(L):
        A_prev = A
        W, b, g = M[layer]
        A, cache = linear_activation_forward(A_prev, W, b, g)
        caches.append(cache)

    assert(A.shape == (1, X.shape[1]))
    
    return A, caches


def compute_cost(AL, Y, g):
    """
    Calculates the cost.

    Args:
        AL (ndarray): probability vector corresponding to "label" predictions (activations of last layer, returned by
                    L_model_forward(), shape 1 by number of examples
        Y (ndarray): true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape 1 by number of examples

    Returns:
    :cost: cross-entropy cost
    
    """
    cost = None
    m = Y.shape[1]

    # only sigmoid or softmax
    assert (g in [f for f in ACTIVATION_FUNCTIONS if f in ['sigmoid', 'softmax']])

    # compute loss from AL and Y
    if g == 'sigmoid':
        cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))) / m
    elif g == 'softmax':
        cost = np.sum(-np.sum(np.multiply(Y, np.log(AL)), axis=0)) / m

    cost = np.squeeze(cost)  # to make sure cost's shape is as expected
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    """
    Linear portion of backward propagation for a single layer (layer l).

    Args:
        dZ (ndarray): gradient of the cost with respect to the linear output (of current layer l)
        cache (tuple): coming from the forward propagation in the current layer

    Returns:
        dA_prev (ndarray): gradient of the cost with respect to the activation (of the previous layer l-1), same shape
                        as A_prev
        dW (ndarray): gradient of the cost with respect to W (current layer l), same shape as W
        db (ndarray): gradient of the cost with respect to b (current layer l), same shape as b
    
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

    Args:
        dA (ndarray): post-activation gradient, numpy array of any shape
        Z (ndarray): pre-activation parameter stored in cache during forward propagation

    Returns:
        dZ (ndarray): gradient of the cost function with respect to Z
    
    """
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def relu_backward(dA, Z):
    """
    Backward propagation for ReLU activation function, vectorized version.

    Args:
        dA (ndarray): post-activation gradient, numpy array of any shape
        Z (ndarray): pre-activation parameter stored in cache during forward propagation

    Returns:
        dZ (ndarray): gradient of the cost function with respect to Z
    
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def lrelu_backward(dA, Z, alpha=0.01):
    """
    Backward propagation for Leaky ReLU activation function, vectorized version.

    Args:
        dA (ndarray): post-activation gradient, numpy array of any shape
        Z (ndarray): pre-activation parameter stored in cache during forward propagation

    Returns:
        dZ (ndarray): gradient of the cost function with respect to Z

    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = alpha

    assert (dZ.shape == Z.shape)

    return dZ


def tanh_backward(dA, Z):
    """
    Backward propagation for tanh activation function, vectorized version.

    Args:
        dA (ndarray): post-activation gradient, numpy array of any shape
        Z (ndarray): pre-activation parameter stored in cache during forward propagation

    Returns:
        dZ (ndarray): gradient of the cost function with respect to Z

    """
    dZ = dA * (1 - np.power(tanh(Z), 2))
    assert (dZ.shape == Z.shape)

    return dZ


def linear_activation_backward(dA, cache, g):
    """
    Backward propagation for the LINEAR->ACTIVATION layer.
    
    Args:
        dA (ndarray): post-activation gradient for current layer l
        cache (tuple): (linear_cache, activation_cache) stored for computing backward propagation efficiently
        g (string): the activation to be used in this layer
    
    Returns:
        dA_prev (ndarray): gradient of the cost with respect to the activation of the previous layer, same shape as A_prev
        dW (ndarray): gradient of the cost with respect to W of current layer, same shape as W
        db (ndarray): gradient of the cost with respect to b of the current layer, same shape as b
    
    """
    dZ, dA_prev, dW, db = None, None, None, None

    # non-implemented activation function (softmax has a different algorithm, presumably used only in output layer)
    assert (g in [f for f in ACTIVATION_FUNCTIONS if f != 'softmax'])

    linear_cache, Z = cache
    
    if g == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif g == "relu":
        dZ = relu_backward(dA, Z)
    elif g == "lrelu":
        dZ = lrelu_backward(dA, Z)
    elif g == 'tanh':
        dZ = tanh_backward(dA, Z)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, G):
    """
    Backward propagation for the model.
    
    Args:
        AL (ndarray): probability vector, output of the forward propagation L_model_forward()
        Y (ndarray): true "label" vector (for example containing 0 if non-cat, 1 if cat)
        caches (list): list of caches (linear and activation) returned from the L_model_forward()
        G (list): list of activation functions for the model

    Returns:
        grads (dict): a dictionary with the gradients

    """
    grads = {}
    Y = Y.reshape(AL.shape)
    L = len(caches)  # the number of layers in the network

    # initializing the backpropagation (output layer must be either sigmoid or softmax)
    assert (G[L] in [f for f in ACTIVATION_FUNCTIONS if f in ['sigmoid', 'softmax']])
    if G[L] == 'sigmoid':
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
            = linear_activation_backward(dAL, caches[L - 1], G[L])
    elif G[L] == 'softmax':
        dZL = AL - Y
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
            = linear_backward(dZL, caches[L-1][0])

    # loop from l=L-1 to l=1
    for layer in reversed(range(1, L)):
        grads["dA" + str(layer-1)], grads["dW" + str(layer)], grads["db" + str(layer)] \
            = linear_activation_backward(grads["dA" + str(layer)], caches[layer-1], G[layer])

    return grads


def update_model(M, grads, learning_rate):
    """
    Update model parameters using gradient descent.
    
    Args:
        M (list): list containing weights and bias parameters, and activation functions for all layers in a network
        grads (dict): dictionary containing gradients, output of L_model_backward()
        learning_rate (float): model's learning rate

    """
    L = len(M)  # number of layers in the neural network

    for layer_id in range(L):
        M[layer_id][0] = M[layer_id][0] - learning_rate * grads["dW" + str(layer_id+1)]
        M[layer_id][1] = M[layer_id][1] - learning_rate * grads["db" + str(layer_id+1)]


def L_layer_model(X, Y, model, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    A L-layer neural network.

    Args:
        X (ndarray): data, numpy array of shape n_x (number of features) by m (number of examples)
        Y (ndarray): true "label" vector (for example containing 0 if cat, 1 if non-cat) of shape 1 by number of examples
        model (tuple): tuple containing model definitions (each element is a tuple containing number of units, activation
                    function and weights initialization method for the layer)
        learning_rate (float): learning rate of the gradient descent update rule
        num_iterations (int): number of iterations of the optimization loop
        print_cost (bool): if True, it prints the cost at every 100 steps

    Returns:
        M (list): list defining the model (each element representing a layer with its learned weights and bias
                parameters, and activation functions) to be used for prediction

    """

    np.random.seed(SEED_VALUE)

    costs = []  # keep track of cost

    # model initialization
    M = initialize_model(X.shape[0], model)

    # loop gradient descent
    for i in range(0, num_iterations):
        # forward propagation
        AL, caches = L_model_forward(X, M)
        # compute cost
        cost = compute_cost(AL, Y, M[-1][2])
        # backward propagation
        grads = L_model_backward(AL, Y, caches, [activation for W, b, activation in M])
        # update model
        update_model(M, grads, learning_rate)

        # print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, float(cost)))
            costs.append(cost)

    return M


def predict(X, M, Y=None):
    """
    Function used to predict the results of a L-layer neural network.

    Args:
        X (ndarray): data set of examples to label, numpy array of shape n_x (number of features) by m (number of examples)
        M (list): definition of the trained model, returned by L_layer_model()
        Y (ndarray): if given, true "label" vector of shape 1 by number of examples to print the accuracy

    Returns:
        P (ndarray): predictions for the given dataset X, shape 1 for sigmoid activated output layer or number of classes
                    for softmax classifier by number of examples

    """
    # forward propagation
    probabilities, _ = L_model_forward(X, M)

    # activation function of the output layer
    g = M[-1][2]
    # only sigmoid and softmax implemented as activation function for the output layer
    assert ([f for f in ACTIVATION_FUNCTIONS if f in ['sigmoid', 'softmax']])

    if g == 'softmax':
        # convert probabilities into classes for all examples
        P = np.argmax(probabilities, axis=0)
    else:
        P = np.where(probabilities > 0.5, 1, 0)

    if Y is not None:
        print("Accuracy: " + str(np.mean(P == Y)))

    return P
