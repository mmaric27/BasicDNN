# Generic L-layer (ReLU-Sigmoid) 'straight in Python' Deep Neural Network implementation using basic Python/numpy.
# Other activation functions, mini-batches, regularization, optimization and batch normalization to be implemented.
#
# usage example:    params = L_layer_model(trainX, trainY, LAYERS_DIMS, num_iterations=1500)
#                   predictTrain = predict(trainX, parameters, trainY)
#                   predictDev = predict(devX, parameters, devY)
#                   predictTest = predict(testX, parameters, testY)

# main package
import numpy as np

# seed global variable
SEED_VALUE = 3

# constants defining the model
LAYERS_DIMS = []  # [12288, 20, 7, 5, 1] example of 4 layer model, 3 hidden + output, 12288 is the # of input features


def sigmoid(Z):
    """
    Implements the sigmoid activation function, vectorized version (array Z).
    
    Params:
    :Z: numpy array of any shape, output of the linear layer
    
    Returns:
    :A: post-activation output of sigmoid(z), same shape as Z
    :cache: cached Z needed for backpropagation

    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache


def relu(Z):
    """
    Implements the ReLU activation function, vectorized version (array Z).

    Params:
    :Z: numpy array of any shape, output of the linear layer

    Returns:
    :A: post-activation output of relu(Z), of the same shape as Z
    :cache: cached Z needed for backpropagation

    """

    A = np.maximum(0, Z)
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for SIGMOID activation function, vectorized version.

    Params:
    :dA: post-activation gradient, numpy array of any shape
    :cache: Z stored earlier for computing backward propagation efficiently

    Returns:
    :dZ: gradient of the cost function with respect to Z
    
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def relu_backward(dA, cache):
    """
    Implements the backward propagation for ReLU activation function, vectorized version.

    Params:
    :dA: post-activation gradient, numpy array of any shape
    :cache: Z stored earlier for computing backward propagation efficiently

    Returns:
    :dZ: gradient of the cost function with respect to Z
    
    """
    
    Z = cache

    dZ = np.array(dA, copy=True)    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def initialize_parameters(layer_dims):
    """
    Initialize weight matrices and bias vectors.
    
    Params:
    :layer_dims: python array (list) containing the dimensions (# of units) 
                 of each layer in network
    
    Returns:
    :parameters: python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                bl -- bias vector of shape (layer_dims[l], 1)

    """
    
    np.random.seed(SEED_VALUE)
    
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for layer in range(1, L):
        parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        parameters['b' + str(layer)] = np.zeros((layer_dims[layer], 1))
        
        assert(parameters['W' + str(layer)].shape == (layer_dims[layer], layer_dims[layer - 1]))
        assert(parameters['b' + str(layer)].shape == (layer_dims[layer], 1))
        
    return parameters


def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation (wa + b), vectorized version.

    Parameters:
    :A: activations from previous layer (input data) of shape number of units of previous layer by number of examples
    :W: weights matrix, numpy array of shape size (# of units) of current layer by size of previous layer
    :b: bias vector, numpy array of shape size of the current layer by 1

    Returns:
    :Z: the input of the activation function, pre-activation parameter 
    :cache: a Python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implements the forward propagation for the LINEAR->ACTIVATION layer.

    Params:
    :A_prev: activations from previous layer (or input data for the first layer) of shape size of previous layer by
            number of examples
    :W: weights matrix, numpy array of shape size of current layer by size of previous layer
    :b: bias vector, numpy array of shape size of the current layer by 1
    :activation: the activation function to be used in layer, stored as a text string: "sigmoid" or "relu" (only those
                two for now)

    Returns:
    :A: the output of the activation function, post-activation value 
    :cache: a Python tuple containing linear cache and activation cache stored for computing the backward pass
            efficiently

    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    else:
        # elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    Implements forward propagation for the layers in a network, for now hard coded [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    (as output layer) computation.
    
    Params:
    :X: data, numpy array of shape input size (# of features) by number of examples
    :parameters: output of initialize_parameters()
    
    Returns:
    :AL: last post-activation value (from the output layer, prediction probability)
    :caches: list of caches containing every cache of linear_activation_forward() (L-1 of them, indexed from 0 to L-1)
    
    """

    caches = []
    A = X
    L = len(parameters)  # number of layers in the network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches


def compute_cost(AL, Y):
    """
    Implements the cost function defined by equation -[y*log(y_hat)+(1-y)*log(1-y_hat)].

    Params:
    :AL: probability vector corresponding to "label" predictions, y_hat in the aforementioned function, shape 1 by
        number of examples
    :Y: true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape 1 by number of examples

    Returns:
    :cost: cross-entropy cost
    
    """
    
    m = Y.shape[1]

    # Compute loss from AL and Y.
    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))) / m
    
    cost = np.squeeze(cost)  # to make sure cost's shape is as expected (fow example, this turns [[17]] into 17).
    
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer (layer l).

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


def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer (for now only relu and sigmoid).
    
    Parameters:
    :dA: post-activation gradient for current layer l 
    :cache: tuple of values (linear_cache, activation_cache) stored for computing backward propagation efficiently
    :activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    :dA_prev: gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :dW: gradient of the cost with respect to W (current layer l), same shape as W
    :db: gradient of the cost with respect to b (current layer l), same shape as b
    
    """
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    else:
        # elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implements the backward propagation for the model, for now hard coded as [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID
        (output layer).
    
    Params:
    :AL: probability vector, output of the forward propagation L_model_forward()
    :Y: true "label" vector (for example containing 0 if non-cat, 1 if cat)
    :caches: list of caches containing:
            every cache of linear_activation_forward() with "relu" it's caches[l], for l in range(L-1) i.e l = 0...L-2)
            the cache of linear_activation_forward() with "sigmoid" it's caches[L-1]
    
    Returns:
    :grads: a dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    
    """
    
    grads = {}
    
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients.
    # Inputs: "dAL, current_cache". 
    # Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] \
        = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    
    # Loop from l=L-2 to l=0
    for layer in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(layer + 1)], current_cache".
        # Outputs: "grads["dA" + str(layer)], grads["dW" + str(layer + 1)], grads["db" + str(layer + 1)]
        current_cache = caches[layer]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(layer+1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(layer)] = dA_prev_temp
        grads["dW" + str(layer + 1)] = dW_temp
        grads["db" + str(layer + 1)] = db_temp
 
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent.
    
    Params:
    :parameters: Python dictionary containing weights and bias parameters 
    :grads: Python dictionary containing gradients, output of L_model_backward
    :learning_rate: model's learning rate
    
    Returns:
    :parameters: Python dictionary containing updated parameters 
                parameters["W" + str(l)] = ... 
                parameters["b" + str(l)] = ...
    
    """
    
    L = len(parameters)  # number of layers in the neural network

    for layer in range(L):
        parameters["W" + str(layer+1)] = parameters["W" + str(layer+1)] - learning_rate * grads["dW" + str(layer+1)]
        parameters["b" + str(layer+1)] = parameters["b" + str(layer+1)] - learning_rate * grads["db" + str(layer+1)]
    
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network with [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID (output layer) activation functions.

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

    np.random.seed(SEED_VALUE)

    costs = []  # keep track of cost

    # parameters initialization
    parameters = initialize_parameters(layers_dims)

    # loop gradient descent
    for i in range(0, num_iterations):
        # forward propagation, for now [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        # compute cost
        cost = compute_cost(AL, Y)
        # backward propagation
        grads = L_model_backward(AL, Y, caches)
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    return parameters


def predict(X, parameters, Y=None):
    """
    Function used to predict the results of a L-layer neural network.

    Params:
    :X: data set of examples to label, numpy array of shape n_x (number of features) by m (number of examples)
    :parameters: parameters of the trained model, returned by L_layer_model()
    :Y: if given, true "label" vector of shape 1 by number of examples to print the accuracy

    Returns:
    :P: predictions for the given dataset X, shape 1 by number of examples

    """

    m = X.shape[1]
    P = np.zeros((1, m))

    # forward propagation
    probabilities, _ = L_model_forward(X, parameters)

    # convert probabilities to 0/1 predictions
    for i in range(0, probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            P[0, i] = 1
        else:
            P[0, i] = 0

    if Y is not None:
        print("Accuracy: " + str(np.sum((P == Y) / m)))

    return P
