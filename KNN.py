# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:30:12 2018

@author: Ashish
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:17:05 2018

@author: Ashish
"""

#-------------------DATA PREPROCESSING-----------------------------------

#Importing the libraries
import numpy as np #Contains mathematical tools and used to include any type of mathematical codes
import matplotlib.pyplot as plt #This allows to plot charts 
import pandas as pd #Import and manage datasets


#Import the datasets using pandas 
dataset = pd.read_csv("Social_Network_Ads.csv")

#Matrix of features. In this X is the independent data and Y is the dependent data.
X = dataset.iloc[:, [2, 3]].values #This line means that we are taking all the columns except the last ome 
Y = dataset.iloc[:,4].values #This line means that we are taking the last column of that is "Purchased" 

#Training and Testing data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature scaling is used to avoid the highly huge or small number to avoid domination of one variable over another. It could be done either with standarisation or normalization

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train) #applying ss_X to the training set, then the data must first be fit then trained 
X_test = ss_X.transform(X_test) #applying ss_X to the testing set, then the data only need to be transformed.


#-----------------KNN MODEL-----------------------------------

from sklearn.neighbors import KNeighborsClassifier
knn_cls = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_cls.fit(X_train, Y_train)

#Prediction of the test set results
Y_pred = knn_cls.predict(X_test)

#Making the confusion Matrix

from sklearn.metrics import confusion_matrix
Conf_Matrix = confusion_matrix(Y_test, Y_pred)

# Visualising the Training set results
"""
Idea behind the graph is to take all the pixels of the frame with 0.01 resolution and applied classifer on it.
It scanned all the pixels and each new pixel points are checked and predicted whether its 0(red) or 1(green) and colourize accoridngly.
Since the logistic regression is a classifer therefore the limits between those sets of point will be a straight line

"""
from matplotlib.colors import ListedColormap 
#Create local variables
X_set, Y_set = X_train, Y_train

#Prepare the grids with all the pixel points. Taking minimum and maximum value of age and salary.(range) with 0.01 resolution 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# With the help of contour function, it uses the classifier and create all the classification into green or red.
plt.contourf(X1, X2, knn_cls.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

#Plotting all the red and green datapoints into the graph using the loop
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Visualization of testing set results 

from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, knn_cls.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


