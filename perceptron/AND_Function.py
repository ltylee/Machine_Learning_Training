"""
Machine Learning Training
AND Function
6/9/2020
https://www.python-course.eu/simple_neural_network.php
"""

import numpy as np
from perceptrons import Perceptron
import matplotlib.pyplot as plt

def labelled_samples(n):
    """
    creates n samples
    """
    for _ in range(n):
        s = np.random.randint(0, 2, (2,))
        yield (s, 1) if s[0] == 1 and s[1] == 1 else (s, 0)

# initialize perceptron
p = Perceptron(weights=[0.3, 0.3, 0.3],
               learning_rate=0.2)

# goes through all the samples and trains the perceptron
for in_data, label in labelled_samples(30):
    p.adjust(label, 
             in_data)

# creates a test set of samples
test_data, test_labels = list(zip(*labelled_samples(30)))

# evaluates perceptron using the test set
evaluation = p.evaluate(test_data, test_labels)
print(evaluation)

# plots the results
fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
X = np.arange(xmin, xmax, 0.1)
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="r")
ax.scatter(1, 1, color="g")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
m = -p.weights[0] / p.weights[1]
c = -p.weights[2] / p.weights[1]
print(m, c)
ax.plot(X, m * X + c )
plt.plot()