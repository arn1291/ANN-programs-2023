import numpy as np
#Define the sigmoid activation function and its derivative
def sigmoid(x):
return 1 / (1+np. exp(- x) )
def sigmoid derivative(x):
return x = (1 - x)
# Define the XOR gate truth table
X = np.array C[ [theta, theta] [0, 1], [1, 0], [1, 1]]) y = np ([[0], [1], [1], [0]])
# Set the random seed for reproducibility
np.random.seed(0)
# Initialize the weights and biases
input size=2
hidden_size=4
output size = 1
learning rate = 0.1
weights_input_hidden= np.random. uniform(size=(input_size, hidden_size)) bias_hid en np.zeros((1, hidden_size))
weights_hidden_output = np.random. uniform (size=(hidden_size, output_size))
bias_output = np.zeros((1, output_size))
#Training the neural network
epochs 10000
for epoch in range(epochs): #Forward propagation
hidden input = np.dot (X, weights_input_hidden) + bias_hidden hidden_output = sigmoid(hidden_input)
output_layer_input = np.dot (hidden_output, weights_hidden_output) bias_output output layer output sigmoid (output_layer_input)
