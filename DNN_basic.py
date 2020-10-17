# Generic L-layer 'straight in Python' Deep Neural Network implementation using basic Python/numpy.
# #Input data is supposed to be stacked in a matrix of n_x by m, where n_x is a number of input features for an example
# and m is the number of training examples.
# Output data is supposed to be stacked in a 1 by m matrix, where m is the number of training examples.
# Output layer can be either Sigmoid or Softmax classifier.
# Implemented activation functions: Sigmoid, ReLU, Leaky ReLU, Tanh, Softmax.
# Implemented weights initialization methods: zeros, random, He, Xavier.
# Implemented regularization methods: L2, Dropout.
# Implemented optimization methods: Mini-Batch Gradient Descent, Momentum, Adam
#
# usage example:    parameters, _ = L_layer_model(trainX, trainY, MODEL)
#                   predictTrain = predict(trainX, parameters, trainY)
#                   predictDev = predict(devX, parameters, devY)
#                   predictTest = predict(testX, parameters, testY)

# main package
import numpy as np

# currently implemented activation functions
ACTIVATION_FUNCTIONS = ('sigmoid', 'relu', 'lrelu', 'tanh', 'softmax')
# currently implemented wights initializations
INITIALIZATIONS = ('zeros', 'random', 'xavier', 'he')
# currently implemented optimization methods
OPTIMIZATIONS = ('gd', 'momentum', 'adam')

# seed global variable
SEED_VALUE = 0

# tuple defining the model (layers dimensions, activation functions and weights initialization method), for
# instance ((20, 'relu', 'he'), (7, 'relu', 'he'), (5, 'relu', 'he'), (1, 'sigmoid', 'random')) is an example of 4 layer
# model, 3 hidden layers with 20, 7, 5 units and relu activation function and he weights initialization, and output
# layer with one unit, sigmoid function and random initialization
MODEL = ()


def one_hot_matrix(Y):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column corresponds to the jth
    training example. So if example j had a label i. Then entry (i,j) will be 1.

    Args:
        Y (ndarray): true "label" vector of shape (1, number of examples)

    Returns:
        one_hot (ndarray): one hot matrix
        
    """
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1

    return one_hot


def initialize_model(n_x, model, optimizer='gd', seed=0):
    """
    Initialize model, its weight matrix, bias vector, activation functions and gradients.
    
    Args:
        n_x (integer): number of input features
        model (tuple): network layer definitions, each element in a tuple is a tuple containing number of units in a
                    layer, activation function and weights initialization method for a layer
        optimizer (string): parameter optimization method when using mini-batches
        seed (int): seed for RandomState

    Returns:
        parameters (list): network layer parameters and activation function
        optimizations (list): list containing parameters for momentum or adam optimization methods, empty if none used

    """
    # unimplemented optimization method
    assert (optimizer in OPTIMIZATIONS)

    parameters = []
    optimizations = []

    np.random.seed(seed)
    
    for layer_id in range(len(model)):
        nodes, g, method = model[layer_id]
        # number of units in a previous layer
        n_prev = model[layer_id-1][0] if layer_id > 0 else n_x

        if method == 'zeros':
            W = np.zeros((nodes, n_prev))
        elif method == 'xavier':
            W = np.random.randn(nodes, n_prev) * np.sqrt(1/n_prev)
        elif method == 'he':
            W = np.random.randn(nodes, n_prev) * np.sqrt(2/n_prev)
        else:
            W = np.random.randn(nodes, n_prev) * 0.01

        b = np.zeros((nodes, 1))

        parameters.append([W, b, g])

        if optimizer != 'gd':
            v_dW = np.zeros((W.shape[0], W.shape[1]))
            v_db = np.zeros((b.shape[0], b.shape[1]))
            # RMSprop part of Adam
            s_dW = np.zeros((W.shape[0], W.shape[1]))
            s_db = np.zeros((b.shape[0], b.shape[1]))

            optimizations.append([v_dW, v_db, s_dW, s_db])

    return parameters, optimizations


def mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Args:
        X (ndarray): input data of shape (input size, number of examples)
        Y (ndarray): true "label" vector of shape (1, number of examples)
        mini_batch_size (int): size of the mini-batches, if 0 using the whole dataset (no Mini-Batch GD optimization)
        seed (int): seed for RandomState

    Returns:
        batches (list): list of synchronous (mini_batch_X, mini_batch_Y)

    """
    batches = []

    if mini_batch_size == 0:
        batches.append((X, Y))
    else:
        np.random.seed(seed)
        # number of training examples
        m = X.shape[1]

        # shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        # number of mini batches of size mini_batch_size in chosen partitioning
        complete_minibatches = int((m / mini_batch_size) // 1)

        # partition (shuffled_X, shuffled_Y), minus the end case
        for k in range(0, complete_minibatches):
            mini_batch_X = shuffled_X[:, mini_batch_size * k: mini_batch_size * (k + 1)]
            mini_batch_Y = shuffled_Y[:, mini_batch_size * k: mini_batch_size * (k + 1)]

            batches.append((mini_batch_X, mini_batch_Y))

        # end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, mini_batch_size * complete_minibatches:]
            mini_batch_Y = shuffled_Y[:, mini_batch_size * complete_minibatches:]

            batches.append((mini_batch_X, mini_batch_Y))

    return batches


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

    assert (Z.shape == (W.shape[0], A.shape[1]))
    
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

    assert (A.shape == Z.shape)

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

    assert (A.shape == Z.shape)

    return A


def lrelu(Z, alpha=0.01):
    """
    Leaky ReLU activation function, vectorized version (array Z).

    Args:
        Z (ndarray): numpy array of any shape, output of the linear layer
        alpha (float): leaky rely alpha value

    Returns:
        A (ndarray): post-activation output of lrelu(Z), same shape as Z

    """
    A = np.maximum(alpha * Z, Z)

    assert (A.shape == Z.shape)

    return A


def tanh(Z):
    """
    Tanh activation function, vectorized version (array Z).

    Args:
        Z (ndarray): numpy array of any shape, output of the linear layer

    Returns:
        A (ndarray): post-activation output of tanh(Z), same shape as Z

    """
    A = (np.exp(Z)-np.exp(-Z)) / (np.exp(Z)+np.exp(-Z))

    assert (A.shape == Z.shape)

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

    assert (A.shape == Z.shape)

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
    # non-implemented activation function
    assert (g in ACTIVATION_FUNCTIONS)

    Z = linear_forward(A_prev, W, b)

    A = eval(g)(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    return A, ((A_prev, W, b), Z)


def L_model_forward(X, parameters, keep_prob=(), seed=0):
    """
    Forward propagation for the layers in a network.
    
    Args:
        X (ndarray): data, array of shape input size (# of features) by number of examples
        parameters (list): output of initialize_model() containing weights and bias parameters and layer activation functions
        keep_prob (tuple): tuple containing probabilities of keeping a neuron active during drop-out for each hidden
                        layer, if empty considered 1 for all layers (keeping all neurons)
        seed (int): seed for RandomState

    Returns:
        A (ndarray): last post-activation value (from the output layer, prediction probability)
        caches (list): list of caches containing every cache of linear_activation_forward()
        dropouts (list): list of dropouts and keep probabilities used during forward propagation for each layer

    """
    caches = []
    dropouts = []

    np.random.seed(seed)

    if keep_prob == ():
        # set probability of keeping neuron for hidden layers to 1 (keeping all neurons)
        keep_prob = (1,) * (len(parameters)-1)
    assert (len(keep_prob) == len(parameters)-1)

    # forward propagation for L layers and add "cache" to the "caches" list
    A = X
    for layer in range(len(parameters)):
        A_prev = A
        W, b, g = parameters[layer]
        A, cache = linear_activation_forward(A_prev, W, b, g)

        # drop-out
        if layer == len(parameters)-1 or keep_prob[layer] == 1:
            # if output layer or not using dropout
            D = np.ones((A.shape[0], A.shape[1]))
        else:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob[layer]).astype(int)
            # shut down some neurons
            A = np.multiply(A, D)
            # scale the value of neurons that haven't been shut down
            A = A / keep_prob[layer]
        dropouts.append((D, keep_prob[layer]))

        caches.append(cache)

    assert (A.shape == (1, X.shape[1]))
    
    return A, caches, dropouts


def compute_cost(AL, Y, parameters, lambd=0):
    """
    Calculates the cost.

    Args:
        AL (ndarray): probability vector corresponding to "label" predictions (activations of last layer, returned by
                    L_model_forward(), shape 1 by number of examples
        Y (ndarray): true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape 1 by number of examples
        parameters (list): output of initialize_model() containing weights and bias parameters and layer activation
                        functions
        lambd (float): L2 regularization hyperparameter, default 0 (no regularization)

    Returns:
        cost (float): cross-entropy cost
    
    """
    cost = None
    # activation function of the output layer in a network
    g = parameters[-1][2]

    # only sigmoid or softmax
    assert (g in [f for f in ACTIVATION_FUNCTIONS if f in ('sigmoid', 'softmax')])

    # compute loss from AL and Y
    if g == 'sigmoid':
        cost = np.sum(np.multiply(-np.log(AL), Y)+np.multiply(-np.log(1-AL), (1-Y)))
    elif g == 'softmax':
        cost = np.sum(np.sum(np.multiply(-np.log(AL), one_hot_matrix(Y)), axis=0))

    if lambd != 0:
        # variable to keep sum of squared weights
        w2_sum = 0
        for layer in range(len(parameters)):
            W, _, _ = parameters[layer]
            w2_sum += np.sum(np.square(W))
        cost += w2_sum * (lambd / 2)

    cost = np.squeeze(cost)

    return float(cost)


def linear_backward(dZ, cache, lambd=0):
    """
    Linear portion of backward propagation for a single layer (layer l).

    Args:
        dZ (ndarray): gradient of the cost with respect to the linear output (of current layer l)
        cache (tuple): coming from the forward propagation in the current layer
        lambd (float): L2 regularization hyperparameter, default 0 (no regularization)

    Returns:
        dA_prev (ndarray): gradient of the cost with respect to the activation (of the previous layer l-1), same shape
                        as A_prev
        dW (ndarray): gradient of the cost with respect to W (current layer l), same shape as W
        db (ndarray): gradient of the cost with respect to b (current layer l), same shape as b
    
    """
    A_prev, W, b = cache

    dW = np.dot(dZ, A_prev.T)
       
    if lambd != 0:
        dW = dW + (lambd / A_prev.shape[1] * W)
        
    db = np.sum(dZ, axis=1, keepdims=True)
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
    S = sigmoid(Z)
    dZ = dA * S * (1 - S)
    
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
        alpha (float): slope of the activation function at x < 0

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


def linear_activation_backward(dA, cache, g, lambd=0):
    """
    Backward propagation for the LINEAR->ACTIVATION layer.
    
    Args:
        dA (ndarray): post-activation gradient for current layer l
        cache (tuple): (linear_cache, activation_cache) stored for computing backward propagation efficiently
        g (string): the activation to be used in this layer
        lambd (float): L2 regularization hyperparameter, default 0 (no regularization)

    Returns:
        dA_prev (ndarray): gradient of the cost with respect to the activation of the previous layer, same shape as
                        A_prev
        dW (ndarray): gradient of the cost with respect to W of current layer, same shape as W
        db (ndarray): gradient of the cost with respect to b of the current layer, same shape as b
    
    """
    linear_cache, Z = cache
    
    # non-implemented activation function (softmax has a different algorithm, presumably used only in output layer)
    assert (g in [f for f in ACTIVATION_FUNCTIONS if f != 'softmax'])

    dZ = eval(g+'_backward')(dA, Z)

    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, dropouts, G, lambd=0):
    """
    Backward propagation for the model.
    
    Args:
        AL (ndarray): probability vector, output of the forward propagation L_model_forward()
        Y (ndarray): true "label" vector (for example containing 0 if non-cat, 1 if cat)
        caches (list): list of caches (linear and activation) returned from the L_model_forward()
        dropouts (list): list of dropouts used during forward propagation, returned from the L_model_forward()
        G (list): list of activation functions for the model
        lambd (float): L2 regularization hyperparameter, default 0 (no regularization)

    Returns:
        grads (list): list with the gradients for each layer

    """
    L = len(caches)  # the number of layers in the network

    # initialize grads with empty lists for each layer, so we can go backward filling it
    grads = []
    for layer in range(L+1):
        grads.append([])

    # activation function of the output layer must be either sigmoid or softmax
    assert (G[L-1] in [f for f in ACTIVATION_FUNCTIONS if f in ['sigmoid', 'softmax']])
    
    # initializing the backpropagation
    dZL = 1./Y.shape[1] * (AL - (one_hot_matrix(Y) if G[L-1] == 'softmax' else Y))
    dA_prev, dW, db = linear_backward(dZL, caches[L-1][0], lambd)
    grads[L] = [None, dW, db]
    grads[L-1] = dA_prev

    # loop from l=L-1 to l=1
    for layer in reversed(range(1, L)):
        dA = grads[layer]
        # applying drop-out on the same neurons it was applied on during forward propagation
        D, keep_prob = dropouts[layer-1]
        if keep_prob != 1:
            dA = np.multiply(dA, D)
            dA = dA / keep_prob

        dA_prev, dW, db = linear_activation_backward(dA, caches[layer-1], G[layer-1], lambd)
        
        grads[layer] = [dA, dW, db]
        grads[layer-1] = dA_prev

    return grads


def update_model(parameters, grads, learning_rate, optimizations, beta=0.9, beta1=0.9, beta2=0.999,  epsilon=1e-8, t=0):
    """
    Update model parameters using gradient descent.
    
    Args:
        parameters (list): list containing weights and bias parameters, and activation functions for all layers in a network
        grads (dict): dictionary containing gradients, output of L_model_backward()
        learning_rate (float): model's learning rate
        optimizations (list): list containing parameters for momentum or adam optimization methods, empty if none used
        beta (float): momentum optimization hyperparameter
        beta1 (float): adam optimization hyperparameter, exponential decay for the past gradients estimates
        beta2 (float): exponential decay hyperparameter for the past squared gradients estimates, adam optimization
                    hyperparameter
        epsilon (float): adam optimization hyperparameter, preventing division by zero
        t (int): bias correction parameter for adam optimization method

    """
    L = len(parameters)  # number of layers in the neural network

    for layer_id in range(L):
        W, b, g = parameters[layer_id]
        _, dW, db = grads[layer_id+1]

        # if optimizations list is empty we're using basic gradient descent and just updating parameters
        if not optimizations:
            W = W - learning_rate * dW
            b = b - learning_rate * db
        else:
            v_dW, v_db, s_dW, s_db = optimizations[layer_id]
            # if there is no RMSprop optimization part (s_dW is all zeros) we're using momentum optimizer
            if not np.any(s_dW):
                # compute velocities
                v_dW = beta * v_dW + (1 - beta) * dW
                v_db = beta * v_db + (1 - beta) * dW
                # update parameters
                W = W - learning_rate * v_dW
                b = b - learning_rate * v_db
            # adam optimization
            else:
                # moving average of the gradients
                v_dW = beta1 * v_dW + (1 - beta1) * dW
                v_db = beta1 * v_db + (1 - beta1) * db
                # moving average of the squared gradients
                s_dW = beta2 * s_dW + (1 - beta2) * np.square(dW)
                s_db = beta2 * s_db + (1 - beta2) * np.square(db)
                # compute bias-corrected first moment estimate
                v_dW_corrected = v_dW / (1 - beta1**t)
                v_db_corrected = v_db / (1 - beta1**t)
                # compute bias-corrected second raw moment estimate
                s_dW_corrected = s_dW / (1 - beta2**t)
                s_db_corrected = s_db / (1 - beta2**t)
                # update parameters
                W = W - learning_rate * (v_dW_corrected / (np.sqrt(s_dW_corrected) + epsilon))
                b = b - learning_rate * (v_db_corrected / (np.sqrt(s_db_corrected) + epsilon))

            optimizations[layer_id] = [v_dW, v_db, s_dW, s_db]

        parameters[layer_id] = [W, b, g]


def L_layer_model(X, Y, model, learning_rate=0.0075, num_epochs=10000, lambd=0, keep_prob=(), mini_batch_size=0,
                  optimizer='gd', beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, print_cost=False):
    """
    A L-layer neural network.

    Args:
        X (ndarray): data, numpy array of shape n_x (number of features) by m (number of examples)
        Y (ndarray): true "label" vector (for example containing 0 if cat, 1 if non-cat) of shape 1 by number of
                    examples
        model (tuple): tuple containing model definitions (each element is a tuple containing number of units,
                    activation function and weights initialization method for the layer)
        learning_rate (float): learning rate of the gradient descent update rule
        num_epochs (int): number of epochs (iterations of the optimization loop)
        lambd (float): L2 regularization hyperparameter, default 0 (no regularization)
        keep_prob (tuple): tuple containing probabilities of keeping a neuron active during drop-out for each hidden
                        layer in a model, if empty considered 1 for all layers (keeping all neurons)
        mini_batch_size (int): the size of a mini batch, if 0 using whole dataset (no Mini-Batch GD optimization)
        optimizer (string): parameter optimization method when using mini-batches
        beta (float): momentum optimization hyperparameter
        beta1 (float): adam optimization hyperparameter, exponential decay for the past gradients estimates
        beta2 (float): exponential decay hyperparameter for the past squared gradients estimates, adam optimization
                    hyperparameter
        epsilon (float): adam optimization hyperparameter, preventing division by zero
        print_cost (bool): if True, it prints the cost at every 100 steps

    Returns:
        parameters (list): list defining the parameters (each element representing a layer with its learned weights and bias
                parameters, and activation functions) to be used for prediction
        costs (list): list of costs for every 100 epochs

    """
    # there should be the same number of examples in X and Y
    assert (X.shape[1] == Y.shape[1])

    costs = []  # keep track of cost
    seed = SEED_VALUE

    # model initialization
    parameters, optimizations = initialize_model(X.shape[0], model, optimizer, seed)
    # there should be the same number of output values (nodes) in output layer of the model and classes in true "label"
    # vector
    assert (parameters[-1][0].shape[0] == Y.max() + (1 if parameters[-1][2] == 'softmax' else 0))

    # initialize bias correction parameter in case adam optimization is used
    t = 0

    # loop gradient descent
    for i in range(0, num_epochs):
        # increasing SEED_VALUE to reshuffle the dataset into mini batches differently after each epoch and to ensure
        # different randomization of dropout neurons in each epoch, if using dropout regularization
        seed += 1
        # define random minibatches
        minibatches = mini_batches(X, Y, mini_batch_size, seed)
        # keep track of total cost for all mini batches
        cost = 0

        for minibatch_X, minibatch_Y in minibatches:
            # forward propagation
            AL, caches, dropouts = L_model_forward(minibatch_X, parameters, keep_prob, seed)
            # compute cost
            cost += compute_cost(AL, minibatch_Y, parameters, lambd)
            # backward propagation
            grads = L_model_backward(AL, minibatch_Y, caches, dropouts, [activation for W, b, activation in parameters], lambd)
            # update model
            update_model(parameters, grads, learning_rate, optimizations, beta, beta1, beta2,  epsilon, t)

        # calculate the cost for the whole set (divide the sum with total number of training examples in the dataset)
        cost /= X.shape[1]

        # print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        # track cost for every 100 epoch
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters, costs


def predict(X, parameters, Y=None):
    """
    Function used to predict the results of a L-layer neural network.

    Args:
        X (ndarray): data set of examples to label, numpy array of shape n_x (number of features) by m (number of
                    examples)
        parameters (list): definition of the trained model, returned by L_layer_model()
        Y (ndarray): if given, true "label" vector of shape 1 by number of examples to print the accuracy

    Returns:
        P (ndarray): predictions for the given dataset X, shape 1 for sigmoid activated output layer or number of
                    classes for softmax classifier by number of examples

    """
    # forward propagation
    probabilities, _, _ = L_model_forward(X, parameters)

    # activation function of the output layer
    g = parameters[-1][2]
    # only sigmoid and softmax implemented as activation function for the output layer
    assert (g in [f for f in ACTIVATION_FUNCTIONS if f in ('sigmoid', 'softmax')])

    if g == 'softmax':
        # convert probabilities into classes for all examples
        P = np.argmax(probabilities, axis=0)
    else:
        P = np.where(probabilities > 0.5, 1, 0)

    if Y is not None:
        print("Accuracy: " + str(np.mean(P == Y)))

    return P
