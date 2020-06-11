"""
Machine Learning Training
simple neural network with 3 layers
6/10/2020
https://www.python-course.eu/training_neural_networks.php
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from neural_networks1 import NeuralNetwork
from collections import Counter

# creats blobs
n_samples = 300
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2, 6], [6, 2]), 
                             random_state=0)
                             
colours = ('green', 'red', 'blue', 'magenta', 'yellow', 'cyan')
fig, ax = plt.subplots()

# visualize blobs
for n_class in range(2):
    ax.scatter(samples[labels==n_class][:, 0], samples[labels==n_class][:, 1], 
               c=colours[n_class], s=40, label=str(n_class))
plt.show()               
               
# create train and test data sets
size_of_learn_sample = int(n_samples * 0.8)
learn_data = samples[:size_of_learn_sample]
test_data = samples[-size_of_learn_sample:]

# create a neural network with two input nodes, two hidden nodes,
# and one output node
simple_network = NeuralNetwork(no_of_in_nodes=2, 
                               no_of_out_nodes=1, 
                               no_of_hidden_nodes=5,
                               learning_rate=0.3)
                               
# train network with samples from training set
for i in range(size_of_learn_sample):
    simple_network.train(learn_data[i], labels[i])

# evaluate network with samples from test set
# because a sigmoid function is used, 0.5 is
# used as the default threshold. using a value
# less than 0.5 allows accounts for undecided classifiers
# for values near 0.5
def evaluate(data, labels, threshold=0.5):
    evaluation = Counter()
    for i in range(len(data)):
        point, label = data[i], labels[i]
        res = simple_network.run(point)
        if threshold < res < 1 - threshold:
            evaluation["undecided"] += 1
        elif label == 1:
            if res >= 1 - threshold:
                evaluation["correct"] += 1
            else:
                evaluation["wrong"] += 1
        elif label == 0:
            if res <= threshold:
                evaluation["correct"] += 1
            else:
                evaluation["wrong"] += 1
    return evaluation

                
res = evaluate(learn_data, labels)
print(res)




