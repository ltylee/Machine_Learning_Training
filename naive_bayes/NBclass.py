"""
Machine Learning Training
Naive Bayes Classifier class
6/23/2020
https://www.python-course.eu/naive_bayes_classifier_introduction.php
"""

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