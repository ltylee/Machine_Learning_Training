"""
Machine Learning Training
Using scikit with two clusters
6/11/2020
https://www.python-course.eu/neural_networks_with_scikit.php
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

npoints = 50
X, Y = [], []
# class 0
X.append(np.random.uniform(low=-2.5, high=2.3, size=(npoints,)) )
Y.append(np.random.uniform(low=-1.7, high=2.8, size=(npoints,)))

# class 1
X.append(np.random.uniform(low=-7.2, high=-4.4, size=(npoints,)) )
Y.append(np.random.uniform(low=3, high=6.5, size=(npoints,)))

learnset = []
learnlabels = []
for i in range(2):
    # adding points of class i to learnset
    points = zip(X[i], Y[i])
    for p in points:
        learnset.append(p)
        learnlabels.append(i)

# creates test points
npoints_test = 3 * npoints
TestX = np.random.uniform(low=-7.2, high=5, size=(npoints_test,)) 
TestY = np.random.uniform(low=-4, high=9, size=(npoints_test,))
testset = []
points = zip(TestX, TestY)
for p in points:
    testset.append(p)

# plots data points
colours = ["b", "r"]
for i in range(2):
    plt.scatter(X[i], Y[i], c=colours[i])
plt.scatter(TestX, TestY, c="g")
plt.show()

# create MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=150, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

# fits MLP
mlp.fit(learnset, learnlabels)
print("Training set score: %f" % mlp.score(learnset, learnlabels))
print("Test set score: %f" % mlp.score(learnset, learnlabels))

print(mlp.classes_)

# predicts test set
predictions = mlp.predict(testset)
print(predictions)
testset = np.array(testset)
testset[predictions==1]

# plots results
colours = ['#C0FFFF', "#FFC8C8"]
for i in range(2):
    plt.scatter(X[i], Y[i], c=colours[i])


colours = ["b", "r"]
for i in range(2):
    cls = testset[predictions==i]
    Xt, Yt = zip(*cls)
    plt.scatter(Xt, Yt, marker="D", c=colours[i])
plt.show()