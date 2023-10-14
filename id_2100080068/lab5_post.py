import numpy as np

X = np.array([0.5, 0.3, 0.7])

W = np.array([[0.2, 0.3, -0.1],
              [-0.5, 0.7, 0.4],
              [0.1, -0.2, 0.6]])
b = np.array([0.1, -0.3, 0.2])
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
Z = np.dot(W, X) + b
A = sigmoid(Z)

print("Activations of the hidden layer:")
print(A)
thresholds = np.array([0.5, 0.6, 0.7])
activated_neurons = A > thresholds
print("Activated neurons:")
print(activated_neurons)
