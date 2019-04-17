# Neural Network written from scratch (no ML packages)
<img src="img/neural_net1.jpg" width="400" > <img src="img/neural_net2.png" width="400" >
## Introduction
I've created my own very simple 1 layer neural network, specialised in binary classification problems. Where the input data-points are multiplied by the weights and a bias is added. The whole thing is summed (weighted-sum) and fed through an activation function (sigmoid is the default). That would be the prediction output. There are no other layers (i.e. hidden layers) involved.
Just for my own understanding of the mathematical side, I wanted to create a neural network from scratch, using plain python code, and without the help of existing ML libraries/packages (e.g. Keras, PyTorch, Scikit-learn ..etc),  which would provide me with ready made model but wouldn't help me understand the inner working parts of the models (so called "working with a black-box"). 
The model is created inside a method (simple_1_layer_classification_NN) that takes the necessary parameters to make a prediction. 


## Getting Started
I would recommend that when you download the code to your local machine, to run it on Spyder IDE. I find Spyder robust and flexible for running different portions of code, displaying: console output, charts, tracking variables/data structures' values among other features. But you're free to use your own preferred IDE/text editor. 


## Dependencies
* numpy
* matplotlib

Python 2 and 3 both work for this. Use [pip](https://pip.pypa.io/en/stable/) to install any dependencies.


## Usage
**Train the network**:
I have provided an example data-set at the end of the Python file, where you can use it to train the network and evaluate its performance. This line, in the Python file is where you can do that: 
```
nn_model.simple_1_layer_classification_NN(X_train, y_train, 2, 10000, learning_rate=0.2)
```
You can experiment with different parameters, such as: `number of epochs` and `learning rate`. 

**Test the network**:
There's also a `predict` method which can be used to predict the test data-set, using the line below, in the Python (after you have trained the network): 
```
nn_model.predict(X_test, y_test)
``` 

## Improvements
As this is an experimental exploration project for building my own intuition of neural networks, inevitably there are many improvements required, that I'm looking to explore and add in future, including:
- Better and more consistent learning
- Whether `input_dimension` is really required as a paramter in the `simple_1_layer_classification_NN`.
- The method ideally need to accept as many layers and nodes as required, passed via the method arguments, to enable a customised neural network.
- Following from the previous point, as such currently the method accepts either `sigmoid` or `relu` as arguments for the network's activation function. However, as the method is only designed to build a 1 layer neural network (meaning it's both the input and output layer), so `relu` would not be adequate to use for the output layer to predict classification problems. It's mainly effective in the hidden layers. Thus it's currently of no use to pass to the method, and therefore expanding the method to allow for adding multiple layers in order to make use of the `relu` argument. 
- The graph plots a maximum of 100 data points of equal distance apart, for the cost value, no matter how many cost data points we have, ensuring the graph doesn't get overwhelmed with too many data points. However, the X axis value limit only goes up to 100. Which doesn't reflect the actual limit of the data the X axis is representing (the frequency of neural network training runs that produces the cost at the end of every run). So changing the X axis limit from 100 to the actual X axis representation data would be good to have (despite the graph only plots 100 data points as the limit).
- It would be ideal to shuffle the input data matrix, in terms of its observations (rows), before starting the Epoch runs (training), just as a measure of avoiding over-fitting. When I tried this, the neural networks accuracy dropped dramatically, which in theory it shouldn't. 
- Look at what other features are required to make an ideal neural network.
- Once all of these improvements are made, we can look into expanding the whole code as a Python module/package, if it proves that it offers anything different from the other existing ML packages.  

## Contributors
Hareth Naji - hazzaldo@hotmail.com


## Thanks
This project is inspired from the YouTube video series:`Beginner Intro to Neural Networks networks`, by `giant_neural_networks` channel, which I used to get a fundamental understanding of a basic neural network from the algebraic, geometrical and coding level. And hence, give me an intuition of how to build a neural network from scratch without the help of out-of-the-box ML libraries and packages.    