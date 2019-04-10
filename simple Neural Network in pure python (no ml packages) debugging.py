# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:29:24 2019

@author: hazzaldo
"""

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


print(' -- All Predictions vs Targets -- \n')
        for i in range(len(output_data_labels)):
            print('Prediction: {}, Target: {}'.format(self.train_predictions[i], output_data_labels[i]))