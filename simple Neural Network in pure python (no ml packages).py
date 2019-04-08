# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:40:03 2019

@author: hazzaldo
"""

# ------- Simple Neural Network written in pure python (No ML packages) ----------
import numpy as np
import matplotlib.pyplot as plt
import os

class NN_classification:
    
    def __init__(self):
        self.dataset_input_matrix = []
        self.num_of_weights = int()
        self.bias = float()
        self.weights = []
        self.training_weighted_sum = float()
        self.test_weighted_sum = float()
        self.chosen_activation = None
        self.train_activation_func_output = float()
        self.test_activation_func_output = float()
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

    """  
     This is a simple 1 layer neural network function, that solves binary classification problems. It takes the following parameters:
         dataset_input_matrix: (type: array) the dataset to feed to the neural network
         output_data_label: (type: integer) the target ouput of the data input sample. The output should be either 0 or 1 as this neural network expects classification problem.
         input_dimension: (type: integer) how many features / datapoints / independant variables per observation.
         epochs: (type: integer) number of times you wish for the neural network to perform forward and back propagation updating the weights and bias. The more epochs performed the more accurate / optimised the prediction will be, but the more time and computation it will take.
         activation_func: (optional - default: 'sigmoid') (type: string) specify the activation function to be used. The default is 'sigmoid' if nothing is specified in this parameter. Although 'relu' is proved quicker for a neural network to learn, it's best to use for the hidden layers, while sigmoid is best used for output layer because of the dealing with probability. Since we only have 1 layer (considered the output layer as well) so sigmoid is best to be the default. At the moment the network only accepts 'relu' or 'sigmoid' function. A potential improvement to add more activation functions in future.
        learning_rate: (optional - default: 0.2) (type: decimal/float) the fraction/percentage of the derivative to use to do weight and bias correction in a single epoch. The higher the fraction/percentage the quicker the weights and bias will be corrected per epoch. But a learning rate that is too high may overshoot in finding the minimum point of error (best optimisation). A good range would be between 0.1 - 0.4
    """
    # --- nueral network structure diagram --- 

    #    O  output prediction
    #   / \   w1, w2, b
    #  O   O  datapoint 1, datapoint 2

    def simple_1_layer_classification_NN(self, dataset_input_matrix, output_data_label, input_dimension, epochs, activation_func='sigmoid', learning_rate=0.2):
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
        # create a random number for our initial bias to begin with.
        self.bias = np.random.rand()
        
        # We perform the training based on the number of epochs specified
        for i in range(epochs):
            # create random index
            ri = np.random.randint(len(self.dataset_input_matrix))
            # Pick random observation vector: pick a random observation vector of independent variables (x) from the dataset matrix
            input_observation_vector = self.dataset_input_matrix[ri]
        
            # Loop through all the independent variables (x) in the observation
            for i in range(len(input_observation_vector)):
                # Weighted_sum: we take each independent variable in the entire observation, add weight to it then add it to the subtotal of weighted sum
                self.training_weighted_sum += input_observation_vector[i] * self.weights[i]
    
            # Add Bias: add bias to weighted sum
            self.training_weighted_sum += self.bias
            
            # Activation: process weighted_sum through activation function
            self.chosen_activation = activation_func
            if activation_func == 'sigmoid':
                self.train_activation_func_output = NN_classification.sigmoid(self.training_weighted_sum)
            else:
                self.train_activation_func_output = NN_classification.relu(self.training_weighted_sum)
        
            # Prediction: Because this is a single layer neural network, so the activation will be the same as the prediction
            pred = self.train_activation_func_output
            
            target = output_data_label
            
            # Cost: we use the error squared function as the cost function to calculate the prediction error margin
            cost = np.square(pred - target)
            
            # Derivative: bringing derivative from cost with respect to each of the network parameters (weights and bias)
            dCost_dPred = 2 * (pred - target)
            if activation_func == 'sigmoid':
                dPred_dWeightSum = NN_classification.sigmoid_derivation(self.training_weighted_sum)
            else:
               dPred_dWeightSum = NN_classification.relu_derivation(self.training_weighted_sum) 
                
            # Bias is just a number on its own added to the formula so its derivative is just 1
            dWeightSum_dB = 1
            # The derivative of the Weighted Sum with respect to each weight is the input data point / independant variable it's multiplied by. 
            # Therefore I simply assigned the input data array to another variable I called 'dWeightedSum_dWeights'
            # to represent the array of the derivative of all the weights involved. I could've used the 'input_sample'
            # array variable itself, but for the sake of readibility, I created a separate variable to represent the derivative of weight.
            dWeightedSum_dWeights = input_observation_vector
            
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

        
    """
    Predict method used to predict the outcome of a new dataset (whether a test dataset or simply a new dataset) using the already trained neural network model.
        Param:
            test_new_observations: (type: 2-dimensional array/list) The input test dataset. Please ensure this a 2-dimensional array/list (i.e. matrix).
    """
    def predict(self, test_new_observations):
        self.test_new_observations = test_new_observations
        # loop through each observation in the matrix
        for observation in self.test_new_observations:
            # Loop through all the independent variables (x) in the observation
            for i in range(len(observation)):
                # Weighted_sum: we take each independent variable in the entire observation, add weight to it then add it to the subtotal of weighted sum
                self.test_weighted_sum += observation[i] * self.weights[i]
            self.test_weighted_sum += self.bias
        
        # Activation: 
        if self.chosen_activation == 'sigmoid':
           self.test_activation_func_output = NN_classification.sigmoid(self.training_weighted_sum)
        else:
            self.test_activation_func_output = NN_classification.relu(self.training_weighted_sum)
        
        # Prediction: Because this is a single layer neural network, so the activation will be the same as the prediction
        pred = self.train_activation_func_output
        return pred
        
"""
The scenario:
It all starts with a farmer. She like to measure everything around her. She was growing some flowers one day and realised she hadn’t measured them. 
So she decided that this day was the perfect day to take out her rulers and take some measurements. 
She has 2 types of flowers: red and blue flowers. She has many flowers. 
She takes out her 2 rulers and lay one horizantally and the other vertically, connecting with the first ruler at a 90 degree angle, like an X and Y graph. 
She plucks 1 petal from the red flower and lays down on her rulers. 
She measures the petal horizontally, along the petal's length, and then vertically along the petal’s width. 
She repeats the same process with the blue flower. She goes on to measure numerous flowers of red and blue colour. 
However, she forgot to note the color of the last measurement. So it looks like her dataset is incomplete.

she decided to use a neural network to workout the unknown flower color. 
A neural network will automate the task that the farmer had to do to workout the color of the flower based on its measurements. 
The advantage of neural network is that it is much faster and more accurate to crunch through numbers and make the right prediction. 
Let’s say we had 10000 flowers, it wouldn’t be practical to graph all of them and try to predict the unknowns. A neural network can do this much faster.  
We will use a simple neural network to predict what color a flower is, just by giving the width and length of its petal. 
"""
from numpy import array
#define array of datset
# each observation vector has 3 datapoints or 3 columns: length, width, and outcome label (0, 1 to represent blue flower and red flower respectively).  
data = array([[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, 0.5, 1],
        [2,   0.5, 0],
        [5.5, 1,   1],
        [1,   1,   0]])

# separate data: split input, output, train and test data.
X_train, y_train, X_test, y_test = data[:6, :-1], data[:6, -1], data[6:, :-1], data[6:, -1]
