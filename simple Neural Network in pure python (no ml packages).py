# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:40:03 2019

@author: hazzaldo
"""

# ------- Simple Neural Network written in pure python (No ML packages) ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class NN_classification:
    
    def __init__(self):
        self.dataset_input_matrix = []
        self.num_of_weights = int()
        self.bias = float()
        self.weights = []
        self.weighted_sum = float()
        self.activation_func_output = float()
        self.dWeightedSum_dWeights = []
        self.dCost_dWeights = []
        self.costs = []
    
    # -- Activation functions --: 
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    # derivative of sigmoid
    def sigmoid_derivation(x): 
        return NN_classification.sigmoid(x) * (1-NN_classification.sigmoid(x))
    
    def relu(x):
        return np.maximum(0.0, x)
    
    def relu_derivation(x):
        if x <= 0:
            return 0
        else:
            return 1

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
  
  Finally to expand this to the final form of the cost function in a Linear Regression model
  cost = 1/m * (sum(model(X)i) - target i )**2)
  1/m because we want to divide by the number of input samples to get the avrage of cost function
  sum - because we want to add up each input multiplied by its weight and finally add the bias at the end, to get the predicted outcome.
  
"""




    """  
     This is a simple 1 layer neural network function, that solves binary classification problems. It takes the following parameters:
         dataset_input_matrix: (type: array) the dataset to feed to the neural network
         output_data_label: (type: integer) the target ouput of the data input sample. The output should be either 0 or 1 as this neural network expects classification problem.
         input_dimension: (type: integer) how many features / datapoints / independant variables per observation.
         epochs: (type: integer) number of times you wish for the neural network to perform forward and back propagation updating the weights and bias. The more epochs performed the more accurate / optimised the prediction will be, but the more time and computation it will take.
         activation_func: (optional - default: 'relu') (type: string) specify the activation function to be used. The default is 'relu' if nothing is specified in this parameter. At the moment the network only accepts 'relu' or 'sigmoid' function. A potential improvement to add more activation functions in future.
        learning_rate: (optional - default: 0.2) (type: decimal/float) the fraction/percentage of the derivative to use to do weight and bias correction in a single epoch. The higher the fraction/percentage the quicker the weights and bias will be corrected per epoch. But a learning rate that is too high may overshoot in finding the minimum point of error (best optimisation). A good range would be between 0.1 - 0.4
    """
    def simple_1_layer_classification_neural_network(dataset_input_matrix, output_data_label, input_dimension, epochs, activation_func='relu', learning_rate=0.2):
        # The number of weights we will use will depend on number of our input sample data. Each data value will get its own weight.
        self.num_of_weights = (len(input_dimension))
        self.dataset_input_matrix = dataset_input_matrix
        # Set initial network parameters (weights & bias):
        # Will initialise the weights to a uniform distribution and ensure the numbers are small close to 0.
        # We need to loop through all the weights to set them to a random value initially.
        for i in range(self.num_of_weights):
            # create random numbers for our initial weights (connections) to begin with. 'rand' method creates small random numbers. 
            w = np.random.rand()
            self.weights.append(w)   
        # create random number for our initial bias to begin with.
        self.bias = np.random.rand()
        
        # We perform the training based on the number of epochs specified
        for i in range(epochs):
            
            # create random index
            ri = np.random.randint(len(self.dataset_input_matrix))
            # Pick random observation vector: pick a random observation vector of independant variables (x) from the dataset matrix
            input_sample_vector = self.dataset_input_matrix[ri]
        
            # Loop through all the independant variables (x) in the vector of input sample
            for i in range(len(input_sample_vector):
                # Weighted_sum: we take each independant variable in the entire vector of input sample, add weight to it then add it to the subtotal of weighted sum
                self.weighted_sum += input_sample_vector[i] * self.weights[i]
    
            # Add Bias: add bias to weighted sum
           self.weighted_sum += self.bias
            
            # Activation: process weighted_sum through activation function
            if activation_func == 'sigmoid':
                self.activation_func_output = sigmoid(self.weighted_sum)
            else:
                self.activation_func_output = relu(self.weighted_sum)
        
            # Prediction: Because this is a single layer neural network, so the activation will be the same as the prediction
            pred = self.activation_func_output
            
            target = output_data_label
            
            # Cost: we use the error squared function as the cost function to calculate the prediction error margin
            cost = np.square(pred - target)
            
            # Derivative: bringing derivative from cost with respect to each of the network parameters (weights and bias)
            dCost_dPred = 2 * (pred - target)
            if activation_func == 'sigmoid':
                dPred_dWeightSum = sigmoid_derivation(self.weighted_sum)
            else:
               dPred_dWeightSum = relu_derivation(self.weighted_sum) 
                
            # Bias is just a number on its own added to the formula so its derivative is just 1
            dWeightSum_dB = 1
            # The derivative of the Weighted Sum with respect to each weight is the input data point / independant variable it's multiplied by. 
            # Therefore I simply assigned the input data array to another variable I called 'dWeightedSum_dWeights'
            # to represent the array of the derivative of all the weights involved. I could've used the 'input_sample'
            # array variable itself, but for the sake of readibility, I created a separate variable to represent the derivative of weight.
            dWeightedSum_dWeights = input_sample_vector
            
            # Derivative chaining all the derivative functions together (chaining rule)
            # Loop through all the weights to workout the derivative of the cost with respect to each weight:
            for dWeightedSum_dWeight in dWeightedSum_dWeights:
                dCost_dWeight = dCost_dPred * dPred_dWeightSum * dWeightedSum_dWeight
                self.dCost_dWeights.append(dCost_dWeight)
    
            dCost_dB = dCost_dPred * dPred_dWeightSum * dWeightSum_dB
            
            # Backpropagation: update the weights and bias according to the derivatives calculated above.
            # In other word we update the parameters of the neural network to correct parameters and therefore 
            # optimise the neural network prediction to be as accurate to the real output as possible
            # We loop through each weight and update it with its derivative with respect to the cost error function value. 
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - learning_rate * self.dCost_dWeights[i]
            self.bias = self.bias - learning_rate * dCost_dB
            
            # for each 100th loop we're going to get a summary of the
            # prediction compared to the actual ouput
            # to see if the prediction is as expected.
            # Anything in prediction above 0.5 should match value 
            # 1 of the actual ouptut. Any prediction below 0.5 should
            # match value of 0 for actual output 
            if i % 100 == 0:
                self.costs.append(cost)
               
        plt.plot(self.costs)
    
        
