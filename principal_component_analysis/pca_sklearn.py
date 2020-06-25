"""
Machine Learning Training
principal component analysis with sklearn
6/24/2020
https://www.python-course.eu/principal_component_analysis.php
"""

from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()


df = pd.DataFrame(data.data, columns=['Alcohol','Malic_acid','Ash','Alcalinity of ash','Magnesium','Total phenols',
                                                 'Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue',
                                                 'OD280/OD315_of_diluted_wines','Proline'])

"""1. Collect the data"""
"""
df = pd.read_table('data\Wine.txt',sep=',',names=['Alcohol','Malic_acid','Ash','Alcalinity of ash','Magnesium','Total phenols',
                                                 'Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue',
                                                 'OD280/OD315_of_diluted_wines','Proline'])"""

target = data.target


"""2. Normalize the data""" 


df = StandardScaler().fit_transform(df)


"""3. Use the PCA and reduce the dimensionality"""

PCA_model = PCA(n_components=2,random_state=42) # We reduce the dimensionality to two dimensions and set the
                                                            # random state to 42
data_transformed = PCA_model.fit_transform(df,target)*(-1) # If we omit the -1 we get the exact same result but rotated by 180 degrees --> -1 on the y axis turns to 1.
                                                            # This is due to the definition of the vectors. We can define a vector a as [-1,-1] and as [1,1] 
                                                            # the lines spanned is the same --> remove the *(-1) and you will see

# Plot the data

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)


ax0.scatter(data_transformed.T[0],data_transformed.T[1])
for l,c in zip((np.unique(target)),['red','green','blue']):
    ax0.scatter(data_transformed.T[0,target==l],data_transformed.T[1,target==l],c=c,label=l)

ax0.legend()



plt.show()