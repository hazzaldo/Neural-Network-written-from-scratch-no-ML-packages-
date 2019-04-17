# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:40:03 2019

@author: hazzaldo
"""

# ------- Simple Neural Network written in pure python (No ML packages) ----------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

class NN_classification:
    
    def __init__(self):
        self.bias = float()
        self.weights = []
        self.chosen_activation_func = None
        self.chosen_cost_func = None
        self.train_average_accuracy = int()
        self.test_average_accuracy = int()
    
    # -- Activation functions --: 
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
        
    def relu(x):
        return np.maximum(0.0, x)
    
    # -- Derivative of activation functions --:
    def sigmoid_derivation(x): 
        return NN_classification.sigmoid(x) * (1-NN_classification.sigmoid(x))
    
    def relu_derivation(x):
        if x <= 0:
            return 0
        else:
            return 1
    
    # -- Squared-error cost function --:
    def squared_error(pred, target):
        return np.square(pred - target)
    
    # -- Derivative of squared-error cost function --:
    def squared_error_derivation(pred, target):
        return 2 * (pred - target)
    

    """  
     This is a simple 1 layer neural network function, that solves binary classification problems. It takes the following parameters:
         dataset_input_matrix: (type: array) the dataset to feed to the neural network
         output_data_label: (type: integer) the target ouput of the data input sample. The output should be either 0 or 1 as this neural network expects classification problem.
         input_dimension: (type: integer) how many features / datapoints / independant variables per observation.
         epochs: (type: integer) number of times you wish for the neural network to perform forward and back propagation updating the weights and bias. The more epochs performed the more accurate / optimised the prediction will be, but the more time and computation it will take.
         activation_func: (optional - default: 'sigmoid') (type: string) specify the activation function to be used. The default is 'sigmoid' if nothing is specified in this parameter. Although 'relu' is proved quicker for a neural network to learn, it's best to use for the hidden layers, while sigmoid is best used for output layer because of the dealing with probability. Since we only have 1 layer (considered the output layer as well) so sigmoid is best to be the default. At the moment the network only accepts 'relu' or 'sigmoid' function. A potential improvement to add more activation functions in future.
        learning_rate: (optional - default: 0.2) (type: decimal/float) the fraction/percentage of the derivative to use to do weight and bias correction in a single epoch. The higher the fraction/percentage the quicker the weights and bias will be corrected per epoch. But a learning rate that is too high may overshoot in finding the minimum point of error (best optimisation). A good range would be between 0.1 - 0.4
        cost_func: (optional - default: 'squared_error') (type: string) the cost function to use in the neural network to calculate the error margin between prediction and target label output.
    """
    # --- neural network structure diagram --- 

    #    O  output prediction
    #   / \   w1, w2, b
    #  O   O  datapoint 1, datapoint 2

    def simple_1_layer_classification_NN(self, dataset_input_matrix, output_data_labels, input_dimension, epochs, activation_func='sigmoid', learning_rate=0.2, cost_func='squared_error'):
        weights = []
        bias = int()
        cost = float()
        costs = []
        dCost_dWeights = []
        chosen_activation_func_derivation = None
        chosen_cost_func = None
        chosen_cost_func_derivation = None
        correct_pred = int()
        incorrect_pred = int()
        
        # store the chosen activation function to use to it later on in the activation calculation section and in the 'predict' method
        # Also the same goes for the derivation section.        
        if activation_func == 'sigmoid':
            self.chosen_activation_func = NN_classification.sigmoid
            chosen_activation_func_derivation = NN_classification.sigmoid_derivation
        elif activation_func == 'relu':
            self.chosen_activation_func = NN_classification.relu
            chosen_activation_func_derivation = NN_classification.relu_derivation
        else:
            print("Exception error - no activation function utilised, in training method", file=sys.stderr)
            return   
            
        # store the chosen cost function to use to it later on in the cost calculation section.
        # Also the same goes for the cost derivation section.    
        if cost_func == 'squared_error':
            chosen_cost_func = NN_classification.squared_error
            chosen_cost_func_derivation = NN_classification.squared_error_derivation
        else:
           print("Exception error - no cost function utilised, in training method", file=sys.stderr)
           return

        # Set initial network parameters (weights & bias):
        # Will initialise the weights to a uniform distribution and ensure the numbers are small close to 0.
        # We need to loop through all the weights to set them to a random value initially.
        for i in range(input_dimension):
            # create random numbers for our initial weights (connections) to begin with. 'rand' method creates small random numbers. 
            w = np.random.rand()
            weights.append(w)
            
        # create a random number for our initial bias to begin with.
        bias = np.random.rand()
        
        '''
        I tried adding the shuffle step, where the matrix is shuffled only in terms of its observations (i.e. rows)
        but this has dropped the accuracy dramaticly, to the point where the 50% range was the best the model can achieve.
        '''
        #input_matrix = dataset_input_matrix
        # shuffle our matrix observation samples, to decrease the chance of overfitting
        #random.shuffle(dataset_input_matrix)
        #input_matrix1 = dataset_input_matrix
        
        # We perform the training based on the number of epochs specified
        for i in range(epochs):
            
            #reset average accuracy with every epoch
            self.train_average_accuracy = 0
            
            for ri in range(len(dataset_input_matrix)): 
            
                # reset weighted sum value at the beginning of every epoch to avoid incrementing the previous observations weighted-sums on top. 
                weighted_sum = 0
                
                input_observation_vector = dataset_input_matrix[ri]
                # Loop through all the independent variables (x) in the observation
                for x in range(len(input_observation_vector)):
                    # Weighted_sum: we take each independent variable in the entire observation, add weight to it then add it to the subtotal of weighted sum
                    weighted_sum += input_observation_vector[x] * weights[x]
    
                # Add Bias: add bias to weighted sum
                weighted_sum += bias
               
                # Activation: process weighted_sum through activation function
                activation_func_output = self.chosen_activation_func(weighted_sum)    
                
                # Prediction: Because this is a single layer neural network, so the activation output will be the same as the prediction
                pred = activation_func_output
    
                # Cost: the cost function to calculate the prediction error margin
                cost = chosen_cost_func(pred, output_data_labels[ri])
                # Also calculate the derivative of the cost function with respect to prediction
                dCost_dPred = chosen_cost_func_derivation(pred, output_data_labels[ri])
    
                # Derivative: bringing derivative from prediction output with respect to the activation function used for the weighted sum.
                dPred_dWeightSum = chosen_activation_func_derivation(weighted_sum)
                    
                # Bias is just a number on its own added to the weighted sum, so its derivative is just 1
                dWeightSum_dB = 1
                
                # The derivative of the Weighted Sum with respect to each weight is the input data point / independant variable it's multiplied by. 
                # Therefore I simply assigned the input data array to another variable I called 'dWeightedSum_dWeights'
                # to represent the array of the derivative of all the weights involved. I could've used the 'input_sample'
                # array variable itself, but for the sake of readibility, I created a separate variable to represent the derivative of each of the weights.
                dWeightedSum_dWeights = input_observation_vector
                
                # Derivative chaining rule: chaining all the derivative functions together (chaining rule)
                # Loop through all the weights to workout the derivative of the cost with respect to each weight:
                for dWeightedSum_dWeight in dWeightedSum_dWeights:
                    dCost_dWeight = dCost_dPred * dPred_dWeightSum * dWeightedSum_dWeight
                    dCost_dWeights.append(dCost_dWeight)
        
                dCost_dB = dCost_dPred * dPred_dWeightSum * dWeightSum_dB
    
                # Backpropagation: update the weights and bias according to the derivatives calculated above.
                # In other word we update the parameters of the neural network to correct parameters and therefore 
                # optimise the neural network prediction to be as accurate to the real output as possible
                # We loop through each weight and update it with its derivative with respect to the cost error function value. 
                for ind in range(len(weights)):
                    weights[ind] = weights[ind] - learning_rate * dCost_dWeights[ind]
       
                bias = bias - learning_rate * dCost_dB
            
                # Compare prediction to target
                error_margin = np.sqrt(np.square(pred - output_data_labels[ri]))
                accuracy = (1 - error_margin) * 100
                self.train_average_accuracy += round(accuracy)
                
                # Evaluate whether guessed correctly or not based on classification binary problem 0 or 1 outcome. So if prediction is above 0.5 it guessed 1 and below 0.5 it guessed incorrectly. If it's dead on 0.5 it is incorrect for either guesses. Because it's no exactly a good guess for either 0 or 1. We need to set a good standard for the neural net model.
                if (error_margin < 0.5) and (error_margin >= 0):
                    correct_pred += 1 
                elif (error_margin >= 0.5) and (error_margin <= 1):
                    incorrect_pred += 1
                else:
                    print("Exception error - 'margin error' for 'predict' method is out of range. Must be between 0 and 1, in training method", file=sys.stderr)
                    return
                
                costs.append(cost)
                
            # Calculate average accuracy from the predictions of all obervations in the training dataset
            self.train_average_accuracy = round(self.train_average_accuracy / len(dataset_input_matrix), 1)
            
    
        # store the final optimised weights to the weights instance variable so it can be used in the predict method.
        self.weights = weights
        
        # store the final optimised bias to the weights instance variable so it can be used in the predict method.
        self.bias = bias
        
        # Print out results 
        print('Average Accuracy: {}'.format(self.train_average_accuracy))
        print('Correct predictions: {}, Incorrect Predictions: {}'.format(correct_pred, incorrect_pred))
        # plot only 100 data points of equal distance apart no matter how many data points are in the costs array.
        plt_costs = None
        if len(costs) <=100:
            plt_costs = costs
        else:
            freq = int(len(costs) / 100)
            plt_costs = costs[::freq]
        plt.plot(plt_costs)
        plt.ylabel = 'costs'
        plt.xlabel = 'training runs'
        plt.show()

        
     
    """
    Predict method used to predict the outcome of a new dataset (whether a test dataset or simply a new dataset) using the already trained neural network model.
        Param:
            test_new_observations: (type: 2-dimensional array/list) The input test dataset. Please ensure this a 2-dimensional array/list (i.e. matrix).
    """
    def predict(self, test_new_input_dataset, output_data_labels):
        num_of_loops = int()
        correct_pred = int()
        incorrect_pred = int()
        activation_func_output = float()
        predictions = []
        # reset average accuracy for test data
        self.test_average_accuracy = 0
        
        # loop through each observation in the matrix
        for ri in range(len(test_new_input_dataset)):
            # for each observation, we need to reset the value of weighted sum so not to increment on top of the previous observations weights sums
            weighted_sum = 0
            observation = test_new_input_dataset[ri]
            # Loop through all the independent variables (x) in the observation
            for i in range(len(observation)):
                # Weighted_sum: we take each independent variable in the entire observation, add weight to it then add it to the subtotal of weighted sum
                weighted_sum += observation[i] * self.weights[i]
            weighted_sum += self.bias
        
            # Activation: 
            activation_func_output = self.chosen_activation_func(weighted_sum)
    
            # Prediction: Because this is a single layer neural network, so the activation will be the same as the prediction
            pred = activation_func_output
            predictions.append(pred)
            
            # Compare prediction to target
            error_margin = np.sqrt(np.square(pred - output_data_labels[ri]))
            accuracy = (1 - error_margin)* 100
            self.test_average_accuracy += accuracy
            num_of_loops +=1
            
            # Evaluate whether guessed correctly or not based on classification binary problem 0 or 1 outcome. So if prediction is above 0.5 it guessed 1 and below 0.5 it guessed incorrectly. If it's dead on 0.5 it is incorrect for either guesses. Because it's no exactly a good guess for either 0 or 1. We need to set a good standard for the neural net model.
            if (error_margin < 0.5) and (error_margin >= 0):
                correct_pred += 1 
            elif (error_margin >= 0.5) and (error_margin <= 1):
                incorrect_pred += 1
            else:
                print("Exception error - 'margin error' for 'predict' method is out of range. Must be between 0 and 1, in predict method", file=sys.stderr)
                return
        
        # calculate average accuracy of predictions
        self.test_average_accuracy = round(self.test_average_accuracy / len(test_new_input_dataset))
        
        # Print out results
        print('Average Accuracy: {}'.format(self.test_average_accuracy))
        print('Correct predictions: {}, Incorrect Predictions: {}'.format(correct_pred, incorrect_pred))
        

        
"""
The scenario:
It all starts with a farmer. She like to measure everything around her. She was growing some flowers one day and realised she hadn’t measured them. 
So she decided that this day was the perfect day to take out her rulers and take some measurements. 
She has 2 types of flowers: red and blue flowers. She has many flowers. 
She takes ou
t her 2 rulers and lay one horizantally and the other vertically, connecting with the first ruler at a 90 degree angle, like an X and Y graph. 
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
#define array of dataset
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

nn_model = NN_classification()

nn_model.simple_1_layer_classification_NN(X_train, y_train, 2, 1000, learning_rate=0.2)

nn_model.predict(X_test, y_test)