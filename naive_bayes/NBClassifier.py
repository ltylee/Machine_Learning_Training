"""
Machine Learning Training
Naive Bayes Classifier class
6/23/2020
https://www.python-course.eu/naive_bayes_classifier_introduction.php
"""

class Classifier:
    
    def __init__(self, *nbclasses):
        self.nbclasses = nbclasses
        
        
    def prob(self, *d, best_only=True):
        
        nbclasses = self.nbclasses
        probability_list = []
        for nbclass in nbclasses:            
            ftrs = nbclass.features
            prob = 1
            for i in range(len(ftrs)):
                prob *= nbclass.probability_value_given_feature(d[i], ftrs[i])
              
            probability_list.append( (prob, nbclass.name) )

        prob_values = [f[0] for f in probability_list]
        prob_sum = sum(prob_values)
        if prob_sum==0:
            number_classes = len(self.nbclasses)
            pl = []
            for prob_element in probability_list:
                pl.append( ((1 / number_classes), prob_element[1]))
            probability_list = pl
        else:
            probability_list = [ (p[0] / prob_sum, p[1])  for p in probability_list]
        if best_only:
            return max(probability_list)
        else:
            return probability_list

class NBclass:        
        def __init__(self, name, *features):
            self.features = features
            self.name = name
            
        def probability_value_given_feature(self, 
                                            feature_value,
                                            feature):
            """
            p_value_given_feature returns the probability p 
            for a feature_value 'value' of the feature  to occurr
            corresponds to P(d_i | p_j)
            where d_i is a feature variable of the feature i
            """
            
            if feature.freq_sum == 0:
                return 0
            else:
                return feature.frequency(feature_value) / feature.freq_sum
                
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