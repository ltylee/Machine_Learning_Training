"""
Machine Learning Training
Using scikit with mnist
6/11/2020
https://www.python-course.eu/neural_networks_with_scikit.php
"""

import pickle
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

with open("../../pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

# unpack data
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

# create MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, ), 
                    max_iter=480, alpha=1e-4,
                    solver='sgd', verbose=10, 
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1)

# train MLP
train_labels = train_labels.reshape(train_labels.shape[0],)
print(train_imgs.shape, train_labels.shape)

mlp.fit(train_imgs, train_labels)
print("Training set score: %f" % mlp.score(train_imgs, train_labels))
print("Test set score: %f" % mlp.score(test_imgs, test_labels))
help(mlp.fit)

# plots results
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()