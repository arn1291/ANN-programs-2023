import numpy as np

# Define the training data and targets
X = np.array([[1, 1, 1, 1],
              [-1, 1, -1, -1],
              [1, 1, 1, -1],
              [1, -1, -1, 1]])
targets = np.array([1, 1, -1, -1])

# Initialize weights to 0
weights = np.zeros(X.shape[1])

# Learning rate
learning_rate = 1

# Number of iterations
max_iterations = 1000

# Perceptron learning algorithm
for iteration in range(max_iterations):
    error_count = 0
    for i in range(len(X)):
        x = X[i]
        target = targets[i]
        predicted = np.dot(x, weights)
        if predicted * target <= 0:
            weights += learning_rate * target * x
            error_count += 1
    if error_count == 0:
        print(f"Converged after {iteration + 1} iterations.")
        break

# Print the learned weights
print("Learned weights:", weights)
