# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:40:03 2019

@author: hazzaldo
"""

# ------- Simple Neural Network written in pure python (No ML packages) ----------

import numpy as np

# Create the neural network function
def NN(m1, m2, w1, w2, b):
    input_weighted_sum = m1 + m2 * w2 + b
    return sigmoid(input_weighted_sum)

# Activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
# create random numbers for our initial weights (connections) and bias to begin with. 'rand' method creates small random numbers. 
w1 = np.random.rand()
w2 = np.random.rand()
b = np.random.rand()

NN(2, 1, w1, w2, b)