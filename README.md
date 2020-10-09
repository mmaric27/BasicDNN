BasicDNN

Generic L-layer (ReLU-Sigmoid) 'straight in Python' Deep Neural Network implementation using basic Python/numpy.

Other activation functions, mini-batches, regularization, optimization and batch normalization to be implemented.

Usage example: 
    - train model:
      params = L_layer_model(trainX, trainY, LAYERS_DIMS, num_iterations=1500)
    - check training set accuracy:
      predictTrain = predict(trainX, parameters, trainY)
    - check dev set accuracy:
       predictDev = predict(devX, parameters, devY)
    ...