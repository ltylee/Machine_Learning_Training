"""
Machine Learning Training
Naive Bayes Classifier Feature class
6/23/2020
https://www.python-course.eu/naive_bayes_classifier_introduction.php
"""

from collections import Counter
import numpy as np

class Feature:
    
    def __init__(self, data, name=None, bin_width=None):
        self.name = name
        self.bin_width = bin_width
        # places data into bins
        if bin_width:
            self.min, self.max = min(data), max(data)
            bins = np.arange((self.min // bin_width) * bin_width, 
                                (self.max // bin_width) * bin_width,
                                bin_width)
            freq, bins = np.histogram(data, bins)
            self.freq_dict = dict(zip(bins, freq))
            self.freq_sum = sum(freq)
        else:
            self.freq_dict = dict(Counter(data))
            self.freq_sum = sum(self.freq_dict.values())
            
        
        
    def frequency(self, value):
        """
        returns number of occurences for a certain feature value or a binned range
        """
        if self.bin_width:
            value = (value // self.bin_width) * self.bin_width
        if value in self.freq_dict:
            return self.freq_dict[value]
        else:
            return 0