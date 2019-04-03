# Simple Neural Network in pure python (no ML packages)

The scenario:

It all starts with farmer. She like to measure everything around her. She was growing some flowers one day and realised she hadn’t measured them. So she decided that this day was the perfect day to take out her rulers and take some measurements. She has 2 types of flowers: red and blue flowers. She has many flowers. 

She takes out her 2 rulers and lay one horizantally and the other vertically, connecting with the first ruler at a 90 degree angle, like an X and Y graph. She plucks 1 petal from the red flower and lays down on her rulers. She measures the petal horizontally, along the petal's length, and then vertically along the petal’s width. She repeats the same process with the blue flower. She goes on to measure numerous flowers of red and blue colour. However, she forgot to note the color of the last measurement. So it looks like her dataset is incomplete.

She has to think about how to solve this problem, to know the color of flower according to its petal measurement. She thought about a number of ways:
1. She could compare the flower petal measurements against the others in the table, and workout that all the red petals come from similar flowers, so therefore their measurements would all be similar. And the same with the blue flowers. So she could compare all other measurements against the measurement with the unknown color and hope she would find some pattern, but there are a lot of numbers and its kind of a pain. 
2. She thought about a better idea, which is to graph all measurements. She then plotted the unknown flower measurement and found that it lies within the red flowers measurements area of the graph. So in this way she doesn’t have to compare the numbers in the table and try to find some pattern, but simply look at the graph to see where the unknown flower measurement lie, to then be able to complete her dataset.
3. She also has a computer, so she decided to use a neural network to workout the unknown flower color. A neural network will automate the task that the farmer had to do to workout the color of the flower based on its measurements. The advantage of neural network is that it is much faster and more accurate to crunch through numbers and make the right prediction. Let’s say we had 10000 flowers, it wouldn’t be practical to graph all of them and try to predict the unknowns. A neural network can do this much faster.  
We will build a simple neural network to predict what color a flower is just by giving the width and length of its petal. 


### Dependencies
* numpy
Python 2 and 3 both work for this. Use [pip](https://pip.pypa.io/en/stable/) to install any dependencies.


