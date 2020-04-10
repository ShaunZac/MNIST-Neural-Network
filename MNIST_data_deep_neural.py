# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:31:54 2020

@author: Shaun Zacharia
"""


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dnn_app_utils_v3 import *

(train_set_x_orig, train_set_y), (test_set_x_orig, test_set_y) = tf.keras.datasets.mnist.load_data()
train_set_y = pd.get_dummies(pd.Series(train_set_y).astype('category')).to_numpy()
test_set_y = pd.get_dummies(pd.Series(test_set_y).astype('category')).to_numpy()

#seeing one of the images
image_index = 101 # You may select anything up to 60,000
#print('even' if(train_set_y[image_index]) else 'odd') # prints the label
#plt.imshow(train_set_x_orig[image_index], cmap='Greys')

train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], 28, 28, 1)
test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], 28, 28, 1)
train_set_y = train_set_y.T 
test_set_y = test_set_y.T

#Taking only the first 500 images
# train_set_x_orig = train_set_x_orig[0:10000, :, :, :]
# test_set_x_orig = test_set_x_orig[0:2000, :, :, :]
# train_set_y = train_set_y[:, 0:10000]
# test_set_y = test_set_y[:, 0:2000]

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

def L_layer_model(X, Y, layers_dims, test_x, test_y, learning_rate = 0.015, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            print("Training accuracy after iteration %i: %f" %(i, predict(X, Y, parameters)))
            print("Test accuracy after iteration %i: %f" %(i, predict(test_x, test_y, parameters)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
layers_dims= [784, 16, 16, 10]
parameters = L_layer_model(train_set_x, train_set_y, layers_dims, test_set_x, test_set_y, num_iterations = 4000, print_cost = True)
print("Final test accuracy: ", predict(test_set_x, test_set_y, parameters))