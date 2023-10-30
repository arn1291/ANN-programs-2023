import numpy as np

# Define input patterns
x1 = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1])
x2 = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1])

# Create a 2D array 'inp' to hold the input patterns
inp = np.array([x1, x2])

# Print the input patterns
print("Input Patterns:")
print(inp)

# Initialize weights with zeros
weight = np.zeros(len(x1))

# Define target values for each pattern
target = np.array([1, -1])

# Update weights using the Hebbian learning rule
print('\nThe modified values in inp:')
for i in range(0, len(inp)):
    c = inp[i] * target[i]
    weight = weight + c

# Print the updated weights
print("\nUpdated Weight:")
print(weight)
