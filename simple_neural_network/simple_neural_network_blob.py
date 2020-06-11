"""
Machine Learning Training
AND Function
6/9/2020
https://www.python-course.eu/simple_neural_network.php
"""


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from perceptrons import Perceptron
import numpy as np

# creates two linearly seperable blobs
n_samples = 250
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9]), 
                             random_state=0)
 
 
# visualises created blobs
colours = ('green', 'magenta', 'blue', 'cyan', 'yellow', 'red')
fig, ax = plt.subplots()


for n_class in range(2):
    ax.scatter(samples[labels==n_class][:, 0], samples[labels==n_class][:, 1], 
               c=colours[n_class], s=40, label=str(n_class))
plt.plot()
plt.show()

# split samples into training and test sets
n_learn_data = int(n_samples * 0.8) # 80 % of available data points
learn_data, test_data = samples[:n_learn_data], samples[-n_learn_data:]
learn_labels, test_labels = labels[:n_learn_data], labels[-n_learn_data:]

# initialize perceptron with weights and learning rate
p = Perceptron(weights=[0.3, 0.3, 0.3],
               learning_rate=0.8)

# train perceptron
for sample, label in zip(learn_data, learn_labels):
    p.adjust(label,
             sample)

# evaluates the perceptron
evaluation = p.evaluate(learn_data, learn_labels)
print(evaluation)

fig, ax = plt.subplots()

# plotting learn data
colours = ('green', 'blue')
for n_class in range(2):
    ax.scatter(learn_data[learn_labels==n_class][:, 0], 
               learn_data[learn_labels==n_class][:, 1], 
               c=colours[n_class], s=40, label=str(n_class))
    
# plotting test data
colours = ('lightgreen', 'lightblue')
for n_class in range(2):
    ax.scatter(test_data[test_labels==n_class][:, 0], 
               test_data[test_labels==n_class][:, 1], 
               c=colours[n_class], s=40, label=str(n_class))


    
X = np.arange(np.max(samples[:,0]))
m = -p.weights[0] / p.weights[1]
c = -p.weights[2] / p.weights[1]
print(m, c)
ax.plot(X, m * X + c )
plt.plot()
plt.show()