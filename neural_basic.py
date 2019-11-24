#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:09:27 2019

@author: shaun
"""

import numpy as np

def sigmoid(a, deriv=False):
    if(deriv==True):
        return a*(1-a)
    return 1/(1+np.exp(-a))

#input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 1, 1],
              [1, 1, 1]])

#output data
Y = np.array([[1],
              [0],
              [0],
              [0]])
    
#np.random.seed(1)
#synapses
syn0 = np.random.random((3, 4))
syn1 = np.random.random((4, 1))

#training
for i in range(1500):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    
    l2_error = l2 - Y
    l2_delta = l2_error*sigmoid(l2, deriv = True)
    
    if(i%150==0):
        print("Error = " + str(np.mean(np.abs(l2_error))))
    
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*sigmoid(l1, deriv = True)
    
    syn1 -= l1.T.dot(l2_delta)
    syn0 -= l0.T.dot(l1_delta)
print("Output after training:"+'\n' + str(l2))