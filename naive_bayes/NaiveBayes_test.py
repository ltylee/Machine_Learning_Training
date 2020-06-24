"""
Machine Learning Training
NaiveBayes test file
6/23/2020
https://www.python-course.eu/text_classification_python.php
"""

from NaiveBayes import  Pool
import os

DClasses = ["clinton",  "lawyer",  "math",  "medical",  "music",  "sex"]

base = "learn/"
p = Pool()
for i in DClasses:
    p.learn(base + i, i)



base = "test/"
for i in DClasses:
    dir = os.listdir(base + i)
    for file in dir:
        res = p.Probability(base + i + "/" + file)
        print(i + ": " + file + ": " + str(res))