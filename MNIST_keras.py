# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:02:13 2020

@author: Shaun Zacharia
"""


import numpy as np
import tensorflow as tf
import pandas as pd
from keras import layers
from keras import losses
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

(X_train_orig, Y_train), (X_test_orig, Y_test) = tf.keras.datasets.mnist.load_data()
Y_train = pd.get_dummies(pd.Series(Y_train).astype('category')).to_numpy()
Y_test = pd.get_dummies(pd.Series(Y_test).astype('category')).to_numpy()

X_train = X_train_orig/255.
X_test = X_test_orig/255.

train_ex = X_train.shape[0]
test_ex = X_test.shape[0]
X_train = X_train.reshape(train_ex, 28, 28, 1)
X_test = X_test.reshape(test_ex, 28, 28, 1)

# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))


def myModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]

    Returns:
    model -- a Model() instance in Keras
    """
      
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(10, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='myModel')    
    
    return model

model = myModel(X_train.shape[1:])
model.compile(loss=losses.categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 5, batch_size = 64)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

preds = model.evaluate(X_test, Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
