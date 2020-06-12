"""
Machine Learning Training
Using scikit perceptron
6/11/2020
https://www.python-course.eu/neural_networks_with_scikit.php
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()

print(iris.data[:3])
print(iris.data[15:18])
print(iris.data[37:40])


# we extract only the lengths and widthes of the petals:
X = iris.data[:, (2, 3)] 

# turn 3 classes into two classes, i.e.
# iris setosa
# not iris setosa
y = (iris.target==0).astype(np.int8)
print(y)

# create perceptron and fit data X and y
p = Perceptron(random_state=42,
              max_iter=10,
              tol=0.001)
p.fit(X, y)

# predictions
values = [[1.5, 0.1], [1.8, 0.4], [1.3,0.2]]

for value in X:
    pred = p.predict([value])
    print([pred])