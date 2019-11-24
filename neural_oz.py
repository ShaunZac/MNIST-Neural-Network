#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:46:11 2019

@author: shaun
"""

import cv2
import numpy as np

def sigmoid(a, deriv=False):
    if(deriv==True):
        return a*(1-a)
    return 1/(1+np.exp(-a))

#input data
img_arr = []
for i in range(18):
    image = (np.array(cv2.imread(str(i)+".png", 0)))
    image = cv2.resize(image, (72, 54))
    image = image.reshape(-1)
    img_arr.append(image)
img_arr = np.array(img_arr)

#output data
y = []
for i in range(18):
    if i < 9:
        y.append([1, 0])
    else:
        y.append([0, 1])
y = np.array(y)

syn0 = np.random.random((3888, 32))
syn1 = np.random.random((32, 2))

for i in range(10000):
    l0 = img_arr
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    
    l2_error = l2 - y
    l2_delta = l2_error*sigmoid(l2, deriv = True)
    
    if(i%1000==0):
        print("Error = " + str(np.mean(np.abs(l2_error))))
#        print(l2_error)
    
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*sigmoid(l1, deriv = True)
    
    syn1 -= l1.T.dot(l2_delta)
    syn0 -= l0.T.dot(l1_delta)
    
print("Output after training:"+'\n' + str(l2))