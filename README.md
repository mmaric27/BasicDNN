BasicDNN

Generic L-layer 'straight in Python' Deep Neural Network implementation using basic Python/numpy.

Implemented activation functions: Sigmoid, ReLU, Leaky ReLU, Tanh, Softmax.
Usage example: LAYERS = [{'dims': [12288, 20, 7, 5, 1], 'activations': [None, 'relu', 'relu', 'relu', 'sigmoid']}
               params = L_layer_model(trainX, trainY, LAYERS, num_iterations=1500)
               predictTrain = predict(trainX, params, activations, trainY)
               predictDev = predict(devX, params, activations, devY)
               predictTest = predict(testX, params, activations, testY) 

Mini-batches, regularization, optimizations and batch normalization to be implemented.
