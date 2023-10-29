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
output_layer_output = sigmoid(output_layer_input)
loss = y - output_layer_output
# Backpropagation
d_output = loss * sigmoid_derivative(output_layer_output)
loss_hidden = d_output. dot(weights_hidden_output.T)
d_hidden = loss_hidden * sigmoid_derivative(hidden_output)
# Update the weights and biases
weights_hidden_output += hidden_output. T.dot(d_output) * learning_rate weights_input_hidden += X.T.dot (d hidden) * learning_rate bias_output += np. sumd_output, axis=0, keepdims=True) * learning_rate bias_hidden += np. sumd_hidden, axis=0, keepdims=True) * learning_rate
# Testing the neural network
# Print the final output
print ("Final Output:")
print(output_layer_output)
# Print the final weights and biases
print("\nFinal Weights and Biases:")
print ("Weights Input to Hidden Layer:") printmeats. put hi Layer:"
print (bias_hidden)
print("Weights Hidden to Output Layer:")
print (weights_hidden_output)
print is up Output Layer:")
