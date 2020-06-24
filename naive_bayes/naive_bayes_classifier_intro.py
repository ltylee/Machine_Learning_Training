"""
Machine Learning Training
Naive Bayes Classifier introductory Exercise
6/23/2020
https://www.python-course.eu/naive_bayes_classifier_introduction.php

This script calculatates the probability that you will arrive on time with one transfer, given that the first train
is n minutes late.
"""

# the tuples consist of (delay time of train1, number of times)


# tuples are (minutes, number of times)
in_time = [(0, 22), (1, 19), (2, 17), (3, 18),
           (4, 16), (5, 15), (6, 9), (7, 7),
           (8, 4), (9, 3), (10, 3), (11, 2)]
too_late = [(6, 6), (7, 9), (8, 12), (9, 17), 
            (10, 18), (11, 15), (12,16), (13, 7),
            (14, 8), (15, 5)]
            
import matplotlib.pyplot as plt

X, Y = zip(*in_time)

X2, Y2 = zip(*too_late)

bar_width = 0.9
plt.bar(X, Y, bar_width,  color="blue", alpha=0.75, label="in time")
bar_width = 0.8
plt.bar(X2, Y2, bar_width,  color="red", alpha=0.75, label="too late")
plt.legend(loc='upper right')
plt.show()

in_time_dict = dict(in_time)
too_late_dict = dict(too_late)

def catch_the_train(min):
    """
    calculates the probability of catching the second train given that the 1st train is min minutes late
    """
    s = in_time_dict.get(min, 0)
    if s == 0:
        return 0
    else:
        m = too_late_dict.get(min, 0)
        return s / (s + m)

for minutes in range(-1, 13):
    print(minutes, catch_the_train(minutes))