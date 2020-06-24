"""
Machine Learning Training
Naive Bayes Classifier for person_data.txt
6/23/2020
https://www.python-course.eu/naive_bayes_classifier_introduction.php
"""

import numpy as np
import NBClassifier
import matplotlib.pyplot as plt

genders = ["male", "female"]
persons = []
with open("person_data.txt") as fh:
    for line in fh:
        # splits the data
        persons.append(line.strip().split())

firstnames = {}
heights = {}
# creates dictionaries based on gener
for gender in genders:
    firstnames[gender] = [ x[0] for x in persons if x[4]==gender]
    heights[gender] = [ x[2] for x in persons if x[4]==gender]
    heights[gender] = np.array(heights[gender], np.int)
    
for gender in ("female", "male"):
    print(gender + ":")
    print(firstnames[gender][:10])
    print(heights[gender][:10])
    

fts = {}
for gender in genders:
    fts[gender] = NBClassifier.Feature(heights[gender], name=gender, bin_width=5)
    print(gender, fts[gender].freq_dict)

# plots distributions of gender with height    
for gender in genders:
    frequencies = list(fts[gender].freq_dict.items())
    frequencies.sort(key=lambda x: x[1])
    X, Y = zip(*frequencies)
    color = "blue" if gender=="male" else "red"
    bar_width = 4 if gender=="male" else 3
    plt.bar(X, Y, bar_width, color=color, alpha=0.75, label=gender)


plt.legend(loc='upper right')
plt.show()

cls = {}
for gender in genders:
    cls[gender] = NBClassifier.NBclass(gender, fts[gender])
    
c = NBClassifier.Classifier(cls["male"], cls["female"])

#checking class
for i in range(130, 220, 5):
    print(i, c.prob(i, best_only=False))
    
# trains classifier based on first names    
fts = {}
cls = {}
for gender in genders:
    fts_names = NBClassifier.Feature(firstnames[gender], name=gender)
    cls[gender] = NBClassifier.NBclass(gender, fts_names)
    
c = NBClassifier.Classifier(cls["male"], cls["female"])

testnames = ['Edgar', 'Benjamin', 'Fred', 'Albert', 'Laura', 
             'Maria', 'Paula', 'Sharon', 'Jessie']
for name in testnames:
    print(name, c.prob(name))
    

print([person for person in persons if person[0] == "Jessie"])