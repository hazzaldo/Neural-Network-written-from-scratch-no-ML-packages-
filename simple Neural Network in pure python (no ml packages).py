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

# We'll label red flowers as 1 and blue flowers as 0

"""
--- Explanation of cost function used and how to calculate derivative of cost function with respect to weights and bias ---
    
 Cost function we will use is the "squared error": (prediction - target)^2.
 The idea is we workout the slope of the graph of the cost function with respect to each weight and the bias
 at a specific value of the each weight (w) and at a specific value of bias (b).
 This is also called the derivative (rate of change) of cost function with respect to each weight and to the bias.
 Once the derivative or slope of graph is worked out, you deduct only a fraction of that rate of change 
 from the concerned weight or the bias, so that we don't overshoot the minimum point of the graph 
 of the cost function with respect to that weight or the bias.
 The weights and bias are thus updated with a fraction of the slope of graph (or derivative) 
 by deducting that fraction from the weights and bias as a correction. 
 This is all done in a process called an Epoch. Once the weights and bias are updated once,
 another forward feed is carried out with the input data and the newly updated weights and bias
 to output another cost function value, this time with a lower error value. 
 When the weights and/or bias are too high, the slope of cost function is a positive number.
 and when the weights and/or bias are too low the slope of cost function is a negative number.
 So what we want to do is subtract a fraction of the slope from weights and/or bias 
 and no matter where the weights and bias start, that will push the prediction directly 
 towards the target value, and with the right amount. Because as weights and bias get 
 closer and closer to the target the slope of the function approaches 0, so our updates
 become smaller and smaller, until we reach the target. 
 We could have a the following formula for updating the weights and bias:
     weight = weight - 0.1 * cost_function_slope(weight)
 Here we're deducting 10% of the cost function slope with respect to the weight
 or deducting 10% of the derivate (rate of change) of cost function with respect 
 to the weight at the particular weight value.
 The same formula can be applied to the bias as well.
    # Formula for slope = rise / run
 The "run" is the different in two points of x axis (or the weight)
 The "rise" is the difference in y axis (or the function) given the difference
 in x axis, so if difference in x is = h. Then Our slope formula would be:
        slope(x) = cost(x+h) - cost(x) / h. 
  This is the general formula to approximate the slope of any function. Note
  h (or difference in x) should be extremely small, almost reaching to 0, to 
  give more accurate slope of a function, therefore the more accurate the 
  approximation is.
  Note also that x+h point on the graph becomes higher on the graph than x point,
  when the slope is a positive direction, and below x point when slope is in a 
  negaive direction. This makes sense because point x is always less than point x+h.
  Therefore in a negative slope cost(x+h) will be less than cost(x) giving a negative slope.
  If the cost(x) = (x-4)^2 . Then the slope(x) = 2*(x-4)
  Here's a lesson that fully explains this: https://www.youtube.com/watch?v=Gvq9sUHPgrc&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So&index=6
  
"""




  

    