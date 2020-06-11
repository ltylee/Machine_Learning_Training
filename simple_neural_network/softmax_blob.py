"""
Machine Learning Training
simple neural network with 3 layers using softmax
6/10/2020
https://www.python-course.eu/softmax.php
"""

from neural_networks_softmax import NeuralNetwork
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

n_samples = 300
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2, 6], [6, 2]), 
                             random_state=0)


colours = ('green', 'red', 'blue', 'magenta', 'yellow', 'cyan')
fig, ax = plt.subplots()


for n_class in range(2):
    ax.scatter(samples[labels==n_class][:, 0], samples[labels==n_class][:, 1], 
               c=colours[n_class], s=40, label=str(n_class))
    
size_of_learn_sample = int(n_samples * 0.8)
learn_data = samples[:size_of_learn_sample]
test_data = samples[-size_of_learn_sample:]

simple_network = NeuralNetwork(no_of_in_nodes=2, 
                               no_of_out_nodes=2, 
                               no_of_hidden_nodes=5,
                               learning_rate=0.3,
                               softmax=True)
                               
labels_one_hot = (np.arange(2) == labels.reshape(labels.size, 1))
labels_one_hot = labels_one_hot.astype(np.float)

for i in range(size_of_learn_sample):
    #print(learn_data[i], labels[i], labels_one_hot[i])
    simple_network.train(learn_data[i], 
                         labels_one_hot[i])
    
print (simple_network.evaluate(learn_data, labels))
