BasicDNN (in development)

Generic L-layer 'straight in Python' Deep Neural Network implementation using basic Python/numpy.

* Input data is supposed to be stacked in a matrix of n_x by m, where n_x is a number of input features for an example and m is the number of training examples.
* Output data is supposed to be stackex in a 1 by mamtrix, where m is the number of training examples.
* Output layer can be either Sigmoid or Softmax classifier.
* Implemented activation functions: Sigmoid, ReLU, Leaky ReLU, Tanh, Softmax.
* Implemented weights initialization methods: zeros, random, He, Xavier.

* Usage example (4 layer model with 3 hidden layers with 20, 7, 5 units and relu activation function and He weights initialization, and output
layer with one unit, sigmoid function and random initialization): 
	- MODEL = ((20, 'relu', 'he'), (7, 'relu', 'he'), (5, 'relu', 'he'), (1, 'sigmoid', 'random'))
	- model = L_layer_model(trainX, trainY, MODEL, num_iterations=1500)
	- predictTrain = predict(trainX, model, trainY)
	- predictDev = predict(devX, model, devY)
	- predictTest = predict(testX, model, testY) 

* Mini-batches, regularization, optimizations and batch normalization to be implemented.

