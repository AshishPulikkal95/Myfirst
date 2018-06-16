# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:48:17 2018

@author: Ashish
"""

#Random forest regression

#Importing the libraries
import numpy as np #Contains mathematical tools and used to include any type of mathematical codes
import matplotlib.pyplot as plt #This allows to plot charts 
import pandas as pd #Import and manage datasets


#Import the datasets using pandas 
dataset = pd.read_csv('Position_Salaries.csv')

#Matrix of features. In this X is the independent data and Y is the dependent data.

X = dataset.iloc[:, 1:2].values #This line means that we are taking all the columns except the last ome 
Y = dataset.iloc[:, 2].values #This line means that we are taking the last column 

#Fitting the model to the dataset
from sklearn.ensemble import RandomForestRegressor
RFR_reg = RandomForestRegressor(n_estimators= 300, random_state = 0)
RFR_reg.fit(X, Y)

# Predicting the Test set results
Y_pred = RFR_reg.predict(6.5) # Since the input must be an array therefore using np.array we transformed it into an array wiht one cell containing 6.5

# Visualising the data
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, RFR_reg.predict(X_grid), color = 'blue')
plt.title('Salary vs Experience (DTC)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()