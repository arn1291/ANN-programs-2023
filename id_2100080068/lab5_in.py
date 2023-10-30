#Binary Sigmoidal
import numpy as np
import matplotlib.pyplot as plt

def binarysigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.linspace(-5, 5, 100)
plt.plot(X, binarysigmoid(X), 'b')
plt.show()
#Bipolar sigmoidal function
import numpy as np
import matplotlib.pyplot as plt

def bipolarsigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

X = np.linspace(-5, 5, 100)
plt.plot(X, bipolarsigmoid(X), 'b')
plt.show()
#Using Binary Sigmoidal Function
import numpy as np
import sympy as sp
import math

x1 = 0.5
x2 = 0.9
x3 = 0.2
x4 = 0.3
b1, b_w = 1, 0.5

# Initializing the learning rate
temp = 0.3

# Initializing the input weights
w1 = 0.2
w2 = 0.3
w3 = -0.6
w4 = -0.1

for i in range(0, 1):
    # Calculating net input for hidden layer z1
    z1 = x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + b1 * b_w

    # Calculating the output using the sigmoidal function
    outz1 = 1 / (1 + math.exp(-z1))
    print('the output of z1 using binary sigmoidal is', outz1)
